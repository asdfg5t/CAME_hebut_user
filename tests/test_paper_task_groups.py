"""CPU-only checks for paper coverage, protocol matching, storage and view semantics."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import io
import csv

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from came_repro import paper_task_groups as groups
from came_repro import paper_task_runtime as runtime
from came_repro.paper_task_io import BufferedTensorBoardWriter,verified_validation_replay


def test_independent_groups_cover_true_remaining_counts_and_shared_control():
    assert groups.counts(groups.jobs_for_group('clip_text'))['executable_jobs']==520
    assert groups.counts(groups.jobs_for_group('clip_image'))['executable_jobs']==60
    assert groups.counts(groups.jobs_for_group('few_shot'))['executable_jobs']==31
    assert groups.counts(groups.jobs_for_group('ablation_reference'))['executable_jobs']==40
    unique={j['id']:j for group in groups.GROUPS for j in groups.jobs_for_group(group)}
    assert sum(j['kind']!='reference_encoder' for j in unique.values())==1020
    assert sum(j['kind']=='reference_encoder' for j in unique.values())==1
    assert sum(j['kind']=='pretraining' for j in unique.values())==10


def test_wiki_grid_and_image_domain_exclusions_are_preserved():
    text=groups.jobs_for_group('clip_text')
    wiki=[j for j in text if j.get('dataset')=='wiki-cs' and j.get('task')=='node']
    assert {(j['split_id'],j['seed']) for j in wiki}=={(a,b) for a in range(20) for b in range(20)}
    assert {j['dataset'] for j in groups.jobs_for_group('clip_image')}=={'ele-fashion','books-nc','sports-copurchase','cloth-copurchase','books-lp'}


def test_few_shot_requires_separate_complete_corpus_and90_episode_configurations():
    jobs=groups.jobs_for_group('few_shot');training=next(j for j in jobs if j['kind']=='pretraining')
    assert training['config']['datasets']==['ogbn-products','books-nc','cloth-copurchase']
    assert training['config']['training']['epochs']==8 and training['config']['training']['max_steps'] is None
    assert set(training['config']['sampling']['seed_limits'])==set(training['config']['datasets'])
    assert groups.counts(jobs)['few_shot_episode_configurations']==90
    assert all(j.get('dependency')==training['id'] for j in jobs if j['kind']=='evaluation')


def test_all_ablations_keep_same_pools_and_surviving_loss_weights():
    from came_repro.variants import expected_loss_weights
    reference=groups.jobs_for_group('ablation_reference')[0]['config']
    for variant in groups.VARIANTS:
        jobs=groups.jobs_for_group(variant)
        own=[j for j in jobs if j.get('variant')==variant]
        assert groups.counts(own)['pretraining']==1 and groups.counts(own)['evaluations']==40
        c=next(j['config'] for j in own if j['kind']=='pretraining')
        assert c['sampling']==reference['sampling'] and c['training']==reference['training'] and c['seed']==42
        expected=expected_loss_weights(reference['model'],variant)
        assert c['model']==expected


def test_new_LP_head_protocol_is_identical_for_CLIP_control_and_ablations():
    recipes=[j['lp_recipe'] for group in ['clip_text','clip_image','ablation_reference',*groups.VARIANTS]
             for j in groups.jobs_for_group(group) if j.get('task')=='link']
    assert recipes and all(r==recipes[0] for r in recipes)
    assert recipes[0]['max_rounds']==30 and recipes[0]['positive_budget_per_round']==512000


def test_pilot_is_only_a_seed_split_slice_and_does_not_shorten_pretraining():
    assert groups.counts(groups.jobs_for_group('clip_text',True))['evaluations']==13
    for group in groups.GROUPS:
        for job in groups.jobs_for_group(group,True):
            if job['kind']=='evaluation':assert job['seed']==0 and job['split_id']==0
            if job['kind']=='pretraining':assert job['config']['training']['epochs']==8


@pytest.mark.parametrize('tensor',[False,True])
def test_CLIP_modality_selection_uses_correct_blocks_for_both_storage_types(tensor):
    x=np.arange(3*1024,dtype=np.float32).reshape(3,1024)
    source=torch.from_numpy(x.copy()) if tensor else x
    np.testing.assert_array_equal(runtime.select_clip_view(source,'clip_text'),x[:,:512])
    np.testing.assert_array_equal(runtime.select_clip_view(source,'clip_image'),x[:,512:])
    with pytest.raises(ValueError):runtime.select_clip_view(source[:,:512],'clip_image')


def test_tensorboard_buffer_preserves_every_scalar_and_immediate_JSONL_CSV_rows():
    from came_repro.experiment import Experiment
    class Writer:
        def __init__(self):self.rows=[];self.flushes=0;self.closed=False
        def add_scalar(self,*args):self.rows.append(args)
        def flush(self):self.flushes+=1
        def close(self):self.closed=True
    raw=Writer();clock=[0.0]
    experiment=Experiment.__new__(Experiment)
    experiment.writer=BufferedTensorBoardWriter(raw,clock=lambda:clock[0])
    experiment.jsonl=io.StringIO();experiment.csvfile=io.StringIO();experiment.csvwriter=None
    rows=[]
    for step,now in enumerate([0.,2.,7.]):
        clock[0]=now;row=dict(step=step,loss=.5/(step+1));rows.append(row);experiment.metric(row)
        assert json.loads(experiment.jsonl.getvalue().splitlines()[-1])==row
    assert list(csv.DictReader(io.StringIO(experiment.csvfile.getvalue())))[-1]['step']=='2'
    assert len(raw.rows)==6 and raw.flushes==1
    experiment.writer.close();assert raw.closed and raw.flushes==2


@pytest.fixture
def replay_fixture(tmp_path):
    run=tmp_path/'validation';run.mkdir()
    valid=dict(source_node=np.array([0,1,2]),target_node=np.array([1,2,3]))
    ranks=np.array([1.,2.,151.]);mrr=float(np.mean(1./ranks))
    np.savez_compressed(run/'validation_link_ranks.npz',**valid,rank=ranks)
    (run/'validation_summary.json').write_text(json.dumps(dict(selected_round=2,validation_mrr=mrr)))
    (run/'status.json').write_text('{"status":"completed_matched_lp_validation"}')
    head=run/'head.pt';torch.save(dict(head={'weight':torch.ones(1)},selected_round=2,validation_mrr=mrr),head)
    audit=dict(dataset='fixture',raw_sha256={'lp-edge-split.pt':'verified_fixture'})
    lock=dict(classifier_checkpoint=str(head),classifier_sha256=groups.fingerprint(head),validation_run_directory=str(run),
        validation_summary_sha256=groups.fingerprint(run/'validation_summary.json'),
        validation_ranks_sha256=groups.fingerprint(run/'validation_link_ranks.npz'),
        data_audit=audit,selected_head_validation_replayed=True,locked_before_test=True,test_used_for_selection=False)
    return lock,valid,audit


def test_complete_selected_head_replay_reuse_is_verified_without_any_GPU_call(replay_fixture,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CPU-only immutable replay verification'))
    lock,valid,audit=replay_fixture
    summary,head=verified_validation_replay(lock,valid,audit)
    assert summary['selected_round']==head['selected_round']==2


@pytest.mark.parametrize('mutation',['head','ranks','official_ids','selected_round'])
def test_validation_replay_reuse_rejects_tampered_or_mismatched_evidence(replay_fixture,mutation):
    lock,valid,audit=replay_fixture;run=Path(lock['validation_run_directory'])
    if mutation=='head':Path(lock['classifier_checkpoint']).write_bytes(b'changed')
    elif mutation=='ranks':(run/'validation_link_ranks.npz').write_bytes(b'changed')
    elif mutation=='official_ids':valid['source_node']=valid['source_node'][::-1]
    else:
        state=torch.load(lock['classifier_checkpoint'],weights_only=False);state['selected_round']=3
        torch.save(state,lock['classifier_checkpoint']);lock['classifier_sha256']=groups.fingerprint(lock['classifier_checkpoint'])
    with pytest.raises((ValueError,AssertionError)):verified_validation_replay(lock,valid,audit)


def test_owned_checkpoint_writer_rejects_escape_and_preserves_old_file_on_limit(tmp_path,monkeypatch):
    from came_repro import storage_budget
    cache=tmp_path/'checkpoints';run=cache/'fixture_paper_tasks__abc_s0_uuid';run.mkdir(parents=True)
    storage=dict(large_checkpoint_root=str(cache),external_cache_high_water_gib=.02,reserve_free_gib=0)
    original_read=runtime.read
    monkeypatch.setattr(runtime,'read',lambda path:storage if Path(path).name=='storage.json' else dict(own_checkpoint_reservation_gib=.02))
    monkeypatch.setattr(storage_budget,'check_budget',lambda **kwargs:dict(external_cache_bytes=0))
    runtime.WRITER_BASELINES.clear()
    target=run/'last.pt';runtime.fast_checkpoint(target,dict(value=torch.ones(3)))
    old=target.read_bytes()
    with pytest.raises(ValueError):runtime.fast_checkpoint(tmp_path/'outside.pt',dict(value=torch.ones(3)))
    monkeypatch.setattr(storage_budget,'tensor_storage_bytes',lambda value:100*2**20)
    with pytest.raises(RuntimeError):runtime.fast_checkpoint(target,dict(value=torch.zeros(3)))
    assert target.read_bytes()==old


def test_completed_artifact_mutation_is_not_reused(tmp_path):
    file=tmp_path/'fixture';file.write_bytes(b'unchanged')
    record=dict(state='completed',artifact_sha256={str(file):groups.fingerprint(file)})
    groups.validate_completed(record)
    file.write_bytes(b'changed')
    with pytest.raises(ValueError):groups.validate_completed(record)


def test_pretraining_capacity_includes_retained_optimizer_and_full_FP32_cache(tmp_path,monkeypatch):
    from came_repro import prepared_data
    (tmp_path/'artifacts/workflow_state').mkdir(parents=True)
    model=tmp_path/'encoder_final.pt';model.write_bytes(b'a'*100)
    (tmp_path/'last.pt').write_bytes(b'a'*300)
    (tmp_path/'artifacts/workflow_state/remaining_main_pilot_current.json').write_text(json.dumps(dict(encoder_checkpoint=str(model))))
    data=tmp_path/'data';data.mkdir();(data/'manifest.json').write_text('{"nodes":685294}')
    monkeypatch.setattr(groups,'ROOT',tmp_path)
    monkeypatch.setattr(prepared_data,'prepared_directory',lambda _:data)
    jobs=[dict(id='new',kind='pretraining'),dict(id='eval',kind='evaluation',representation='came',dataset='books-nc',dependency='new')]
    r=groups.storage_reservation(jobs,{})
    assert r['largest_complete_embedding_bytes']==685294*1024*4+2**20
    assert r['additional_cache_bytes']>685294*1024*4+400
    assert groups.storage_reservation(jobs,{'new':dict(state='completed')})['additional_cache_bytes']==0


def test_owned_cache_eviction_waits_for_all_results_and_never_touches_protected_cache(tmp_path,monkeypatch):
    encoder=tmp_path/'encoder_fixture.pt';encoder.write_bytes(b'fixture-not-model')
    digest=groups.fingerprint(encoder);scope=tmp_path/'owned';cache=scope/digest[:20]/'ele-fashion';cache.mkdir(parents=True)
    np.save(cache/'fused_embeddings.npy',np.ones((3,4),np.float32))
    (cache/'inference_state.json').write_text(json.dumps(dict(complete=True,identity=dict(checkpoint_sha256=digest),output_sha256=groups.fingerprint(cache/'fused_embeddings.npy'))))
    protected=tmp_path/'protected_main_cache';protected.mkdir();(protected/'valuable').write_bytes(b'preserve')
    proof=tmp_path/'result_fixture';proof.write_bytes(b'complete')
    job=dict(id='a',kind='evaluation',representation='came',dependency='encoder',dataset='ele-fashion')
    next_job=dict(job,id='b')
    done=dict(state='completed',artifact_sha256={str(proof):groups.fingerprint(proof)})
    ledger=dict(records={'encoder':dict(encoder_checkpoint=str(encoder)),'a':copy.deepcopy(done)})
    monkeypatch.setattr(runtime,'owned_cache_root',lambda:scope)
    monkeypatch.setattr(groups,'REGISTRY',tmp_path/'registry.json')
    groups.evict_owned_cache(job,ledger,[job,next_job]);assert cache.exists()
    ledger['records']['b']=copy.deepcopy(done)
    groups.evict_owned_cache(next_job,ledger,[job,next_job]);assert not cache.exists()
    assert (protected/'valuable').read_bytes()==b'preserve' and encoder.exists()


@pytest.mark.parametrize('first_code',[75,1])
def test_new_group_resumes_into_new_run_and_reuses_complete_records(tmp_path,monkeypatch,first_code):
    from came_repro import ide_workflow,quality_pipeline,storage_budget
    from came_repro.pause_control import TrainingPaused
    release=tmp_path/'release.json';release.write_text('{}')
    out=tmp_path/'out';registry=out/'shared_jobs.json'
    job=dict(id='fixture_node',kind='evaluation',task='node',dataset='fixture',representation='clip_text',
        variant='clip_text',pretraining_seed=None,seed=0,split_id=0)
    monkeypatch.setattr(groups,'ROOT',tmp_path);monkeypatch.setattr(groups,'OUT',out)
    monkeypatch.setattr(groups,'REGISTRY',registry);monkeypatch.setattr(groups,'RELEASE',release)
    monkeypatch.setattr(groups,'verify_release',lambda:{})
    monkeypatch.setattr(groups,'jobs_for_group',lambda *args:[copy.deepcopy(job)])
    monkeypatch.setattr(sys,'argv',['fixture'])
    class Lock:
        def close(self):pass
    monkeypatch.setattr(ide_workflow,'acquire_gpu_lock',lambda *args:Lock())
    monkeypatch.setattr(storage_budget,'check_budget',lambda:{})
    calls=[]
    def child(arguments,pause,ledger,ledger_path,record,logger,remaining):
        calls.append(arguments);run=tmp_path/('child'+str(len(calls)))
        (run/'checkpoints').mkdir(parents=True)
        (run/'checkpoints/location.json').write_text(json.dumps(dict(checkpoint_directory=str(run/'checkpoints'))))
        record.setdefault('attempts',[]).append(dict(run_directory=str(run),elapsed_active_seconds=3))
        ledger['active_seconds']+=3
        if len(calls)==1:
            (run/'checkpoints/probe_last.pt').write_bytes(b'controller fixture, not model')
            if first_code==75:pause.write_text('{}')
            return first_code,str(run)
        assert arguments[arguments.index('--resume')+1]==str(tmp_path/'child1/checkpoints/probe_last.pt')
        return 0,str(run)
    monkeypatch.setattr(quality_pipeline,'run_child',child)
    def commit(job,record,run):
        proof=Path(run)/'complete_fixture.json';proof.write_text('{"real_metric":false}')
        record.update(state='completed',run_directory=str(run),summary={},artifact_sha256={str(proof):groups.fingerprint(proof)})
    monkeypatch.setattr(groups,'commit',commit)
    with pytest.raises(TrainingPaused if first_code==75 else RuntimeError):groups.run_group('clip_text')
    assert groups.read(registry)['active_seconds']==3
    groups.run_group('clip_text')
    assert groups.read(out/'clip_text/state.json')['state']=='completed_full_group'
    assert groups.read(registry)['active_seconds']==6
    assert groups.read(out/'clip_text/measured_timing.json')['completed_jobs'][0]['active_seconds']==6
    groups.run_group('clip_text')
    assert len(calls)==2 and groups.read(registry)['active_seconds']==6
