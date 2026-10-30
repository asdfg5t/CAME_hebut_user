from pathlib import Path
import copy
import pytest
from came_repro import full_suite as suite
from came_repro import pause_control
from came_repro.ide_workflow import read_json, write_json


def grid():
    profile = read_json(suite.ROOT / 'configs/full_suite_resumable.json')
    return profile, suite.build_jobs(profile, read_json(suite.ROOT / profile['training_config']))


def test_full_grid_has_all_domains_repeats_and_all_nine_ablations():
    profile, jobs = grid()
    assert len(jobs)==1471
    assert sum(job['kind']=='pretraining' for job in jobs)==11
    evaluations=[job for job in jobs if job['kind']=='evaluation']
    wiki=[job for job in evaluations if job['representation']=='came' and job['variant']=='came' and job['dataset']=='wiki-cs' and job['task']=='node']
    assert {(job['split_id'],job['seed']) for job in wiki}=={(split,seed) for split in range(20) for seed in range(20)}
    assert len({job['dataset'] for job in evaluations})==10
    assert sum(job['variant'] in profile['ablation_variants'] for job in evaluations)==360
    few=[job for job in jobs if job['kind']=='pretraining' and job['protocol']=='few_shot'][0]
    assert few['config']['datasets']==['ogbn-products','books-nc','cloth-copurchase']
    assert few['config']['sampling']['seed_limits']['books-nc']==16384
    assert all(job['config']['training']['max_steps'] is None for job in jobs if job['kind']=='pretraining')


def test_resume_uses_latest_coherent_head_checkpoint_even_if_last_attempt_only_inferred(tmp_path):
    profile, jobs = grid()
    job=next(item for item in jobs if item['kind']=='evaluation' and item['task']=='node')
    old=tmp_path/'old'
    new=tmp_path/'new'
    for directory in (old,new):
        directory.mkdir()
        (directory/'checkpoints').mkdir()
        write_json(directory/'manifest.json',{'checkpoint_directory':str(directory/'checkpoints')})
    checkpoint=old/'checkpoints/probe_last.pt'
    checkpoint.write_bytes(b'coherent_fixture')
    records={job['id']:{'attempts':[{'run_directory':str(old)},{'run_directory':str(new)}]},
             job['dependency']:{'state':'completed','encoder_checkpoint':'verified_encoder.pt'}}
    args=suite.job_arguments(job,profile,tmp_path,records)
    assert args[args.index('--resume-head')+1]==str(checkpoint)
    assert args[args.index('--epochs')+1]=='2000'
    records[job['dependency']]['state']='paused'
    with pytest.raises(ValueError,match='Incomplete encoder'):
        suite.job_arguments(job,profile,tmp_path,records)


def test_partial_subset_run_cannot_complete_full_grid_training(tmp_path):
    write_json(tmp_path/'summary.json',{'complete_declared_subset_pretraining':False,'completed_subset_rounds':2})
    write_json(tmp_path/'status.json',{'status':'completed_partial_budget'})
    with pytest.raises(ValueError,match='Partial pretraining'):
        suite.commit_completion({'kind':'pretraining'}, {}, str(tmp_path))


def test_changed_completed_result_is_rejected_instead_of_skipped(tmp_path):
    path=tmp_path/'result.json'
    path.write_text('original')
    record={'artifact_sha256':{str(path):suite.fingerprint(path)}}
    suite.validate_completed(record)
    path.write_text('changed')
    with pytest.raises(ValueError,match='Completed artifact changed'):
        suite.validate_completed(record)


def test_pause_is_scoped_to_suite_environment_and_missing_parent_is_safe(monkeypatch,tmp_path):
    monkeypatch.delenv('CAME_PAUSE_FILE',raising=False)
    monkeypatch.delenv('CAME_SUITE_PARENT_PID',raising=False)
    assert not pause_control.pause_requested()
    path=tmp_path/'pause.json'
    monkeypatch.setenv('CAME_PAUSE_FILE',str(path))
    assert not pause_control.pause_requested()
    path.write_text('{}')
    with pytest.raises(pause_control.TrainingPaused):
        pause_control.pause_if_requested()
    path.unlink()
    monkeypatch.setenv('CAME_SUITE_PARENT_PID','12345')
    import psutil
    monkeypatch.setattr(psutil,'pid_exists',lambda pid:False)
    assert pause_control.pause_requested()


def test_cache_eviction_waits_for_all_seeds_and_preserves_results(tmp_path):
    checkpoint=tmp_path/'encoder.pt'
    checkpoint.write_bytes(b'this_suite_encoder_fixture')
    root=tmp_path/'cache'
    digest=suite.fingerprint(checkpoint)
    target=root/digest[:20]/'dataset'
    target.mkdir(parents=True)
    write_json(target/'inference_state.json',{'identity':{'checkpoint_sha256':digest},'complete':True,'output_sha256':'fixture'})
    (target/'fused_embeddings.npy').write_bytes(b'derived_fixture')
    job={'id':'seed0','kind':'evaluation','dependency':'pretrain','dataset':'dataset'}
    next_job={**job,'id':'seed1'}
    ledger={'records':{'pretrain':{'encoder_checkpoint':str(checkpoint)},'seed0':{'state':'completed'}}}
    suite.evict_finished_embedding_cache(job,[job,next_job],ledger,{'embedding_cache_root':str(root)})
    assert target.exists()
    ledger['records']['seed1']={'state':'completed'}
    suite.evict_finished_embedding_cache(next_job,[job,next_job],ledger,{'embedding_cache_root':str(root)})
    assert not target.exists() and checkpoint.exists()
    assert len(ledger['cache_evictions'])==1


def test_statistics_use_completed_seeds_keep_splits_and_mark_partial_grid():
    _,jobs=grid()
    selected=[job for job in jobs if job['kind']=='evaluation' and job['dataset']=='wiki-cs' and job['task']=='node' and job['variant']=='came'][:3]
    records={job['id']:{'state':'completed','job':job,'summary':{'test_accuracy':value}} for job,value in zip(selected,(.6,.7,.8))}
    result=suite.aggregate_completed(records,jobs)[0]
    assert result['completed_jobs']==3 and result['expected_jobs']==400
    assert not result['complete_requested_grid'] and not result['paper_reproduction_complete']
    assert result['per_split']['0']['test_accuracy']['mean']==pytest.approx(.7)
    assert result['per_split']['0']['test_accuracy']['sample_std']==pytest.approx(.1)


@pytest.mark.parametrize('first_code',[75,1])
def test_pipeline_restart_passes_resume_keeps_elapsed_budget_and_skips_completed_jobs(tmp_path,monkeypatch,first_code):
    import sys
    root=tmp_path/'project'
    (root/'configs').mkdir(parents=True)
    (root/'src/came_repro').mkdir(parents=True)
    (root/'src/came_repro/fixture.py').write_text('CONTROL_FIXTURE=1')
    profile={'training_config':'configs/training.json','max_active_hours':24}
    write_json(root/'configs/full_suite_resumable.json',profile)
    write_json(root/'configs/training.json',{})
    write_json(root/'configs/storage.json',{'embedding_cache_root':str(root/'cache')})
    train={'id':'training_fixture','kind':'pretraining','config':{'name':'fixture'}}
    evaluate={'id':'evaluation_fixture','kind':'evaluation','task':'node','dataset':'dataset','representation':'came',
              'variant':'came','pretraining_seed':42,'seed':0,'split_id':0,'dependency':train['id']}
    profile.update(probe_epochs=2000,probe_patience=100,inference_batch_size=32,products_inference_batch_size=4)
    write_json(root/'configs/full_suite_resumable.json',profile)
    monkeypatch.setattr(suite,'ROOT',root)
    monkeypatch.setattr(suite,'build_jobs',lambda *args:[train,evaluate])
    monkeypatch.setattr(suite,'verify_originals',lambda:{'fixture':'protected_originals'})
    class Lock:
        def close(self):
            pass
    monkeypatch.setattr(suite,'acquire_gpu_lock',lambda *args:Lock())
    monkeypatch.setattr(sys,'executable',str(root/'.venv-came39/Scripts/python.exe'))
    monkeypatch.setattr(sys,'argv',['run','--profile',str(root/'configs/full_suite_resumable.json')])
    calls=[]
    def child(arguments,pause_file,ledger,ledger_path,record,logger,limit_seconds):
        ledger['active_seconds']+=3
        if 'preflight_reproduction.py' in arguments[0]:
            return 0,None
        calls.append(arguments)
        run=root/'children'/str(len(calls))
        (run/'checkpoints').mkdir(parents=True)
        write_json(run/'manifest.json',{'checkpoint_directory':str(run/'checkpoints')})
        write_json(run/'config.json',{'fixture':True})
        record.setdefault('attempts',[]).append({'run_directory':str(run),'arguments':arguments})
        if len(calls)==1:
            (run/'checkpoints/last.pt').write_bytes(b'coherent_controller_fixture_not_model')
            if first_code==75:
                write_json(pause_file,{'reason':'fixture_safe_pause'})
            return first_code,str(run)
        if 'came_repro.subset_offline_train' in arguments:
            assert '--resume' in arguments
            assert arguments[arguments.index('--resume')+1]==str(root/'children/1/checkpoints/last.pt')
            write_json(run/'summary.json',{'complete_declared_subset_pretraining':True,'completed_subset_rounds':8})
            write_json(run/'status.json',{'status':'completed_subset_pretraining'})
            (run/'checkpoints/encoder_final.pt').write_bytes(b'completed_controller_fixture_not_model')
        else:
            write_json(run/'summary.json',{'test_accuracy':.5})
            write_json(run/'status.json',{'status':'completed_downstream_task'})
        return 0,str(run)
    monkeypatch.setattr(suite,'run_child',child)
    assert suite.main()==(0 if first_code==75 else 1)
    pointer=read_json(root/'artifacts/workflow_state/full_suite_current.json')
    paused=read_json(pointer['ledger'])
    assert paused['records'][train['id']]['state']==('paused' if first_code==75 else 'failed') and paused['active_seconds']==6
    suite.main()
    completed=read_json(pointer['ledger'])
    assert completed['state']=='completed_came_task_grid' and completed['active_seconds']==15
    assert len(calls)==3
    suite.main()
    assert len(calls)==3  # preflight runs again; scientific jobs are retained
    assert read_json(pointer['ledger'])['active_seconds']==18
