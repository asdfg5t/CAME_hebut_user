"""Click-to-run plans must preserve grids and reject partial/wrong-corpus encoders."""
import json
from pathlib import Path
import sys

import pytest
from came_repro.ide_workflow import ROOT, CORPORA, completed_encoder, evaluation_jobs, stage_jobs, validate_workflow, main
from came_repro.ide_workflow import acquire_gpu_lock


def workflow():
    return json.loads((ROOT/'configs/pycharm_workflow.json').read_text())


def test_click_evaluation_plan_covers_ten_seeds_and_twenty_wiki_splits():
    config=workflow()
    jobs=evaluation_jobs(config,'came',checkpoint='completed_encoder.pt')
    assert len(jobs)==290
    wiki=[job for job in jobs if job['dataset']=='wiki-cs']
    assert {(job['seed'],job['split_id']) for job in wiki}=={(seed,split) for seed in range(10) for split in range(20)}
    assert len(evaluation_jobs(config,'came',True,checkpoint='few_encoder.pt'))==30
    assert all('--checkpoint' in job['arguments'] and '--max-steps' not in job['arguments'] for job in jobs)
    image=evaluation_jobs(config,'clip_image')
    assert len(image)==50
    assert all(job['dataset'] in ('ele-fashion','books-nc','sports-copurchase','cloth-copurchase','books-lp') for job in image)


def test_click_training_defaults_keep_eight_epoch_full_budget():
    for stage in ('pretrain_representation','pretrain_few_shot','ablation'):
        job=stage_jobs(stage,workflow(),True)[0]
        arguments=job['arguments']
        profile=json.loads(Path(arguments[arguments.index('--config')+1]).read_text())
        assert profile['training']['epochs']==8
        assert profile['training']['max_steps'] is None
        assert all(value is None for value in profile['sampling']['seed_limits'].values())
        assert '--max-steps' not in arguments


def create_run(root,name,protocol,epochs=8,variant='came',seed=42,test_accuracy=0):
    run=root/'runs'/name
    (run/'checkpoints').mkdir(parents=True)
    (run/'checkpoints/encoder_final.pt').write_bytes(b'metadata_selection_fixture')
    documents={'config.json':{'datasets':CORPORA[protocol],'variant':variant,'seed':seed},
               'summary.json':{'completed_epochs':epochs,'complete_declared_pretraining':epochs==8,'test_accuracy':test_accuracy},
               'status.json':{'status':'completed_pretraining' if epochs==8 else 'completed_partial_budget'},
               'manifest.json':{'created_utc':name,'checkpoint_directory':str(run/'checkpoints')}}
    for filename,value in documents.items():
        (run/filename).write_text(json.dumps(value))
    return run


def test_click_encoder_selection_ignores_partial_ablation_wrong_corpus_and_test_score(tmp_path):
    create_run(tmp_path,'01_complete','representation',test_accuracy=.99)
    newest=create_run(tmp_path,'02_complete','representation',test_accuracy=.1)
    create_run(tmp_path,'99_partial','representation',epochs=0,test_accuracy=1)
    create_run(tmp_path,'98_wrong_corpus','few_shot')
    create_run(tmp_path,'97_ablation','representation',variant='without_moe')
    create_run(tmp_path,'96_wrong_seed','representation',seed=5)
    assert completed_encoder('representation',workflow(),tmp_path)==(newest/'checkpoints/encoder_final.pt').resolve()


def test_click_evaluation_refuses_to_use_existing_pilot(tmp_path):
    create_run(tmp_path,'partial','representation',epochs=0)
    with pytest.raises(RuntimeError,match='Partial pilots are not accepted'):
        completed_encoder('representation',workflow(),tmp_path)


@pytest.mark.parametrize('field,values',[('evaluation_seeds',[0,0]),('wiki_split_ids',[20]),('evaluation_seeds',[])])
def test_click_workflow_rejects_invalid_seed_split_grid(field,values):
    config=workflow()
    config[field]=values
    with pytest.raises(ValueError):
        validate_workflow(config)


def test_click_entry_refuses_wrong_python_instead_of_silent_fallback(monkeypatch):
    monkeypatch.setattr(sys,'argv',['run','--dry-run'])
    monkeypatch.setattr(sys,'executable',str(ROOT/'wrong_interpreter.exe'))
    with pytest.raises(RuntimeError,match='Set the PyCharm run interpreter'):
        main('pretrain_representation')


def test_click_gpu_lock_prevents_concurrent_stages_and_releases(tmp_path):
    path=tmp_path/'gpu.lock'
    first=acquire_gpu_lock(path)
    try:
        with pytest.raises(RuntimeError,match='Another CAME PyCharm GPU stage'):
            acquire_gpu_lock(path)
    finally:
        first.close()
    second=acquire_gpu_lock(path)
    second.close()


def test_batch_skips_verified_completion_but_rejects_changed_artifact(tmp_path,monkeypatch):
    from came_repro import ide_workflow,storage_budget
    root=tmp_path/'project'
    (root/'configs').mkdir(parents=True)
    (root/'src/came_repro').mkdir(parents=True)
    (root/'src/came_repro/fixture.py').write_text('FIXTURE_VERSION=1')
    (root/'configs/storage.json').write_text('{}')
    config_path=root/'configs/workflow.json'
    config_path.write_text(json.dumps(workflow()))
    monkeypatch.setattr(ide_workflow,'ROOT',root)
    monkeypatch.setattr(sys,'executable',str(root/'.venv-came39/Scripts/python.exe'))
    monkeypatch.setattr(sys,'argv',['run','--workflow-config',str(config_path)])
    monkeypatch.chdir(root)
    monkeypatch.setattr(storage_budget,'check_budget',lambda *args,**kwargs:{'within_budget':True})
    monkeypatch.setattr(ide_workflow,'stage_jobs',lambda *args:[{'arguments':['fixture_job']}])
    child=root/'runs/child'
    (child/'figures').mkdir(parents=True)
    (child/'status.json').write_text(json.dumps({'status':'completed_downstream_task'}))
    (child/'summary.json').write_text(json.dumps({'task':'node','test_accuracy':.7}))
    artifact=child/'figures/fixture.png'
    artifact.write_bytes(b'fixture_numeric_artifact_v1')
    calls=[]
    def run_fixture(arguments,logger):
        calls.append(arguments)
        return child
    monkeypatch.setattr(ide_workflow,'run_child',run_fixture)
    ide_workflow.main('baselines')
    ide_workflow.main('baselines')
    assert len(calls)==1
    artifact.write_bytes(b'changed_artifact')
    with pytest.raises(ValueError,match='artifact changed'):
        ide_workflow.main('baselines')
    assert len(calls)==1
