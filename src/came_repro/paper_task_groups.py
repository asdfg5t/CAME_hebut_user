"""Independent, hash-pinned remaining paper groups; never trusts legacy completion flags."""
import argparse
import copy
import datetime
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path

from .author_loader import ROOT, fingerprint, verify_originals
from .atomic_io import atomic_write_json

PROFILE=ROOT/'configs/remaining_paper_tasks_20261021.json'
RELEASE=ROOT/'artifacts/delivery/remaining_paper_tasks_release.json'
OUT=ROOT/'artifacts/remaining_paper_tasks'
REGISTRY=OUT/'shared_jobs.json'
VARIANTS=('without_text_vision_loss','without_text_fusion_loss','without_vision_fusion_loss',
    'only_text_vision_loss','only_text_fusion_loss','only_vision_fusion_loss',
    'without_moe','without_dimension_gate','without_modality_gnn')
GROUPS=('clip_text','clip_image','few_shot','ablation_reference')+VARIANTS


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_release():
    verify_originals()
    for file,digest in read(RELEASE)['input_sha256'].items():
        if fingerprint(file)!=digest:raise ValueError('Frozen remaining-task input changed: '+file)
    return read(PROFILE)


def jobs_for_group(group,pilot=False):
    from .full_suite import build_jobs
    from .joint_train import validate_joint
    if group not in GROUPS:raise ValueError('Unknown independent paper group')
    p=read(PROFILE);legacy=read(ROOT/p['legacy_grid_profile']);training=read(ROOT/p['training_config'])
    all_jobs=build_jobs(legacy,training)
    reference=next(copy.deepcopy(j) for j in all_jobs if j['kind']=='pretraining' and j['protocol']=='representation' and j['variant']=='came')
    reference['kind']='reference_encoder'
    controls=[reference]+[copy.deepcopy(j) for j in all_jobs if j['kind']=='evaluation' and j['variant']=='came' and j['task'] in ['node','link'] and j['dataset'] in ['ele-fashion','books-nc','cloth-copurchase','sports-copurchase']]
    if group in ['clip_text','clip_image']:
        selected=[copy.deepcopy(j) for j in all_jobs if j['kind']=='evaluation' and j['representation']==group]
    elif group=='few_shot':
        selected=[copy.deepcopy(j) for j in all_jobs if j['kind']=='pretraining' and j['protocol']=='few_shot']
        selected += [copy.deepcopy(j) for j in all_jobs if j['kind']=='evaluation' and j['representation']=='came' and j['task']=='few_shot']
    elif group=='ablation_reference':selected=controls
    else:selected=controls+[copy.deepcopy(j) for j in all_jobs if j.get('variant')==group]
    for job in selected:
        if job['kind']=='pretraining':
            job['config']['deviations'].append('Independent paper groups; no warm-start; all eight declared subset rounds; exact author subset sizes unreleased.')
            validate_joint(job['config'])
        if job.get('task')=='link':job['lp_recipe']=copy.deepcopy(p['lp_recipe'])
    if pilot:
        selected=[j for j in selected if j['kind']!='evaluation' or (j['seed']==0 and j['split_id']==0)]
    if len({j['id'] for j in selected})!=len(selected):raise ValueError('Duplicate scientific job identity')
    return selected


def counts(jobs):
    return dict(pretraining=sum(j['kind']=='pretraining' for j in jobs),evaluations=sum(j['kind']=='evaluation' for j in jobs),
        reused_reference_encoders=sum(j['kind']=='reference_encoder' for j in jobs),
        executable_jobs=sum(j['kind']!='reference_encoder' for j in jobs),
        few_shot_episode_configurations=3*sum(j.get('task')=='few_shot' for j in jobs))


def validate_completed(record):
    if record['state']!='completed':raise ValueError('An unfinished job cannot be reused')
    for file,digest in record['artifact_sha256'].items():
        if fingerprint(file)!=digest:raise ValueError('Completed paper result changed: '+file)


def commit(job,record,run):
    import numpy as np
    from .prepared_data import prepared_directory
    directory=Path(run);s=read(directory/'summary.json')
    status=read(directory/'status.json')['status']
    expected_status='completed_subset_pretraining' if job['kind']=='pretraining' else 'completed_paper_lp_test' if job.get('task')=='link' else 'completed_downstream_task'
    if status!=expected_status:raise ValueError('Unfinished run cannot be committed as a complete scientific task')
    paths=[directory/x for x in ['summary.json','config.json','manifest.json','status.json','logs/train.log','logs/metrics.csv','logs/metrics.jsonl']]
    if job['kind']=='pretraining':
        if not s.get('complete_declared_subset_pretraining') or s.get('completed_subset_rounds')!=8:
            raise ValueError('Eight complete declared subset rounds required; partial training retained')
        checkpoint=Path(read(directory/'checkpoints/location.json')['checkpoint_directory'])/'encoder_final.pt'
        if not checkpoint.exists():raise ValueError('Final encoder missing')
        record['encoder_checkpoint']=str(checkpoint);paths.append(checkpoint)
    elif job['task']=='link':
        if s.get('test_evaluations')!=1:raise ValueError('Exactly one complete test required')
        import torch
        raw=ROOT/'data/raw/mm_graph'/job['dataset']/'lp-edge-split.pt'
        split=torch.load(str(raw),map_location='cpu',weights_only=False,mmap=True)
        with np.load(directory/'test_link_ranks.npz') as v:
            for key in ['source_node','target_node']:np.testing.assert_array_equal(v[key],split['test'][key])
            ranks=v['rank'];assert len(ranks)==len(split['test']['source_node'])
            measured=float(np.mean(1./ranks.astype(np.float64)))
            if measured!=s['test']['official']['mrr']:raise ValueError('Independent official MRR mismatch')
            for k in [1,3,10]:
                if float(np.mean(ranks<=k))!=s['test']['official']['hits@%d'%k]:raise ValueError('Independent official Hits mismatch')
            from .task_metrics import reverse_overlap_mask
            nodes=read(prepared_directory(job['dataset'])/'manifest.json')['nodes']
            mask=reverse_overlap_mask(np.asarray(split['train']['source_node']),np.asarray(split['train']['target_node']),
                np.asarray(split['test']['source_node']),np.asarray(split['test']['target_node']),nodes)
            np.testing.assert_array_equal(mask,v['reverse_overlap'])
    elif job['task']=='few_shot':
        labels=np.load(prepared_directory(job['dataset'])/'labels.npy',mmap_mode='r').reshape(-1)
        for row in s['episodes']:
            with np.load(directory/('episode_%dshot.npz'%row['shot'])) as v:
                if np.intersect1d(v['support_ids'],v['query_ids']).size:raise ValueError('Support/query leakage')
                if v['support_ids'].shape!=(5,row['shot']):raise ValueError('Wrong5-way support budget')
                np.testing.assert_array_equal(labels[v['support_ids']],np.repeat(v['classes'][:,None],row['shot'],axis=1))
                if not np.isin(labels[v['query_ids']],v['classes']).all():raise ValueError('Queries contain unrelated classes')
                if float(np.mean(v['prediction']==labels[v['query_ids']]))!=row['accuracy']:raise ValueError('Episode accuracy mismatch')
    else:
        with np.load(directory/'test_predictions.npz') as v:
            measured=float(np.mean(v['prediction']==v['labels']))
            if measured!=s['test_accuracy'] or s['test_evaluations']!=1:raise ValueError('Independent accuracy mismatch')
            prepared=prepared_directory(job['dataset'])
            if job['task']=='relation':
                triples=np.load(prepared/'test_triples.npy',mmap_mode='r')
                np.testing.assert_array_equal(v['pairs'],triples[:,[0,2]])
                np.testing.assert_array_equal(v['labels'],triples[:,1])
            else:
                labels=np.load(prepared/'labels.npy',mmap_mode='r').reshape(-1)
                ids=np.flatnonzero(np.load(prepared/'test_mask_indices.npy',mmap_mode='r')) if job['dataset']=='wiki-cs' else np.load(prepared/'test_indices.npy',mmap_mode='r')
                ids=ids[labels[ids]>=0]
                np.testing.assert_array_equal(v['node_ids'],ids)
                np.testing.assert_array_equal(v['labels'],labels[ids])
    paths+=list(directory.glob('*.npz'))+list((directory/'figures').glob('*.png'))+list((directory/'tensorboard').glob('*'))
    location=directory/'checkpoints/location.json'
    if location.exists():paths+=[location]+list(Path(read(location)['checkpoint_directory']).glob('*.pt'))
    if s.get('validation_run_directory'):
        valid=Path(s['validation_run_directory'])
        paths += [valid/x for x in ['validation_summary.json','selection_lock.json','config.json','manifest.json','status.json','logs/train.log','logs/metrics.jsonl','logs/metrics.csv']]
        paths += list((valid/'tensorboard').glob('*'))+list((valid/'figures').glob('*.png'))
        paths += [valid/'checkpoints/location.json']
        paths += list(Path(read(valid/'checkpoints/location.json')['checkpoint_directory']).glob('*.pt'))
    record.update(state='completed',run_directory=str(directory),summary=s,
        artifact_sha256={str(p):fingerprint(p) for p in paths if p.is_file()})


def evict_owned_cache(job,registry,planned):
    """Only this release's derived cache, only after every dependent result exists."""
    import shutil
    from .paper_task_runtime import owned_cache_root
    if job.get('representation')!='came' or job['kind']!='evaluation':return
    required=[j for j in planned if j.get('dependency')==job['dependency'] and j.get('dataset')==job['dataset'] and j['kind']=='evaluation']
    if not required or not all(registry['records'].get(j['id'],{}).get('state')=='completed' for j in required):return
    for item in required:validate_completed(registry['records'][item['id']])
    encoder=registry['records'][job['dependency']]['encoder_checkpoint'];digest=fingerprint(encoder)
    scope=owned_cache_root().resolve();target=(scope/digest[:20]/job['dataset']).resolve()
    if not target.is_relative_to(scope) or len(target.relative_to(scope).parts)!=2 or target.name not in ['ele-fashion','books-nc','sports-copurchase','cloth-copurchase','ogbn-arxiv','wiki-cs']:
        raise ValueError('Derived cache deletion escaped release scope')
    if not target.exists():return
    state=read(target/'inference_state.json')
    if not state['complete'] or state['identity']['checkpoint_sha256']!=digest or fingerprint(target/'fused_embeddings.npy')!=state['output_sha256']:
        raise ValueError('Only a matching complete derived cache may be evicted')
    audit=dict(directory=str(target),encoder_sha256=digest,embedding_sha256=state['output_sha256'],
        dependent_jobs=[j['id'] for j in required],reason='All complete official predictions and classifier checkpoints retained; scoped regenerable cache only')
    registry.setdefault('cache_evictions',[]).append(audit)
    atomic_write_json(REGISTRY,registry)  # Persist the exact resolved target BEFORE deletion.
    shutil.rmtree(target)


def reference_encoder(job,record):
    import torch
    pointer=read(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')
    checkpoint=Path(pointer['encoder_checkpoint'])
    if fingerprint(checkpoint)!=pointer['encoder_sha256']:raise ValueError('Protected reference encoder changed')
    saved=torch.load(str(checkpoint),map_location='cpu',weights_only=False,mmap=True)
    if saved.get('completed_subset_rounds')!=8:raise ValueError('Reference did not complete eight subset rounds')
    for key in ['datasets','seed','model','sampling','training']:
        if saved['config'][key]!=job['config'][key]:raise ValueError('Reference scientific recipe differs: '+key)
    record.update(state='completed',encoder_checkpoint=str(checkpoint),summary=dict(reused_reference=True,completed_subset_rounds=8),
        artifact_sha256={str(checkpoint):fingerprint(checkpoint)})


def storage_reservation(jobs,records):
    """Known reference footprint plus largest live derived cache; never raises limits."""
    new_models=[j for j in jobs if j['kind']=='pretraining' and records.get(j['id'],{}).get('state')!='completed']
    if not new_models:return dict(additional_cache_bytes=0,unmeasured_future_checkpoint_sizes=True)
    from .prepared_data import prepared_directory
    pointer=read(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')
    reference=Path(pointer['encoder_checkpoint'])
    # Final FP32 encoder and full optimizer/RNG last checkpoint are retained.
    checkpoint_bytes=reference.stat().st_size+(reference.parent/'last.pt').stat().st_size
    expected_nodes=[read(prepared_directory(j['dataset'])/'manifest.json')['nodes'] for j in jobs
        if j.get('representation')=='came' and j['kind']=='evaluation' and j.get('dependency') in {x['id'] for x in new_models}]
    largest_cache=max(expected_nodes,default=0)*1024*4+2**20
    required=int(len(new_models)*checkpoint_bytes*1.10)+largest_cache+128*2**20
    return dict(additional_cache_bytes=required,retained_new_models=len(new_models),reference_checkpoint_bytes=checkpoint_bytes,
        largest_complete_embedding_bytes=largest_cache,checkpoint_margin=1.10,head_and_temporary_reserve_bytes=128*2**20,
        estimate_is_not_a_guarantee=True,unmeasured_future_checkpoint_sizes=True)


def summarize(group,registry,selected,full_jobs,directory):
    from .full_suite import aggregate_completed
    records={j['id']:registry['records'][j['id']] for j in full_jobs if j['id'] in registry['records']}
    complete=sum(r['state']=='completed' for r in records.values())
    stats=dict(group=group,completed_shared_records=complete,full_expected=counts(full_jobs),
        selected_expected=counts(selected),mode='full' if len(selected)==len(full_jobs) else 'single_seed_diagnostic',
        aggregate=aggregate_completed(records,full_jobs),paper_reproduction_complete=False)
    atomic_write_json(directory/'results.json',stats)
    actual=[]
    for job in selected:
        record=registry['records'].get(job['id'],{})
        elapsed=sum(a.get('elapsed_active_seconds',0.) for a in record.get('attempts',[]))
        if record.get('state')=='completed' and job['kind']!='reference_encoder':
            actual.append(dict(job=job['id'],kind=job['kind'],task=job.get('task'),dataset=job.get('dataset'),active_seconds=elapsed))
    by_type={}
    for row in actual:
        key=(row['kind'],row['task'],row['dataset']);by_type.setdefault(key,[]).append(row['active_seconds'])
    estimates=[]
    for key,values in by_type.items():
        pending=sum(j['kind']==key[0] and j.get('task')==key[1] and j.get('dataset')==key[2]
            and registry['records'].get(j['id'],{}).get('state')!='completed' for j in selected)
        estimates.append(dict(kind=key[0],task=key[1],dataset=key[2],completed_samples=len(values),pending=pending,
            measured_mean_active_seconds=sum(values)/len(values),estimated_remaining_seconds=pending*sum(values)/len(values),
            includes_unmeasured_inference_or_cold_PPR=False,not_a_guarantee=True))
    atomic_write_json(directory/'measured_timing.json',dict(completed_jobs=actual,same_type_estimates=estimates,
        unmeasured_tasks_have_no_fake_duration=True))


def run_group(group):
    from .ide_workflow import acquire_gpu_lock
    from .quality_pipeline import run_child,best_resume
    from .pause_control import TrainingPaused
    parser=argparse.ArgumentParser(description='Independent '+group+' paper task group')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--pause',action='store_true');parser.add_argument('--max-hours',type=float)
    args=parser.parse_args();p=verify_release()
    directory=OUT/(group+('_pilot' if args.pilot else ''))
    directory.mkdir(parents=True,exist_ok=True)
    pause=directory/'pause.json';ledger_path=directory/'state.json'
    if args.pause:atomic_write_json(pause,dict(reason='user_requested_group_pause'));return
    jobs=jobs_for_group(group,args.pilot);full=jobs_for_group(group)
    if args.dry_run:
        print(json.dumps(dict(group=group,selected=counts(jobs),full=counts(full),pilot=args.pilot,
            actual_GPU_started=False,actual_timing_unknown=True,plan=jobs,unreleased_details=p['unreleased_details']),ensure_ascii=False,indent=2),flush=True);return
    if args.max_hours is not None and args.max_hours<=0:raise ValueError('Positive explicit runtime budget required')
    own=acquire_gpu_lock(ROOT/'.runtime/remaining_paper_tasks.lock');workflow=None
    logger=logging.getLogger('remaining_paper_'+group);logger.setLevel(logging.INFO)
    handlers=[logging.StreamHandler(),logging.FileHandler(directory/'pipeline.log',encoding='utf-8')]
    for h in handlers:
        f=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s');f.converter=time.gmtime;h.setFormatter(f);logger.addHandler(h)
    registry=read(REGISTRY) if REGISTRY.exists() else dict(release_sha256=fingerprint(RELEASE),active_seconds=0.,records={})
    ledger=read(ledger_path) if ledger_path.exists() else dict(group=group,pilot=args.pilot,state='ready',active_seconds=0.,records={},max_active_hours=args.max_hours)
    if registry.get('current_group')==directory.name and ledger.get('state')=='running':
        ledger['active_seconds']+=max(0,registry['active_seconds']-ledger.get('last_global_active_seconds',registry['active_seconds']))
    sleep_guard=False;initial_global=registry['active_seconds'];initial_group=ledger['active_seconds']
    def persist():
        ledger['active_seconds']=initial_group+registry['active_seconds']-initial_global
        ledger['records']={j['id']:registry['records'][j['id']] for j in jobs if j['id'] in registry['records']}
        ledger['last_global_active_seconds']=registry['active_seconds']
        atomic_write_json(REGISTRY,registry);atomic_write_json(ledger_path,ledger)
    try:
        if registry['release_sha256']!=fingerprint(RELEASE):raise ValueError('Shared paper registry has different frozen science/source identity')
        if args.max_hours is not None:
            if ledger['max_active_hours'] is not None and args.max_hours!=ledger['max_active_hours']:raise ValueError('Do not silently change cumulative group budget')
            ledger['max_active_hours']=args.max_hours
        if pause.exists():
            history=directory/'pause_history';history.mkdir(exist_ok=True)
            atomic_write_json(history/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),read(pause));pause.unlink()
        while workflow is None:
            if pause.exists():raise TrainingPaused('Paused while waiting; no GPU task started')
            try:workflow=acquire_gpu_lock(ROOT/'.runtime/goodreads_official.lock')
            except RuntimeError:
                if ledger['state']!='waiting_for_active_workflow':logger.info('WAIT: 保留当前Goodreads/Products流程，不插队。')
                ledger['state']='waiting_for_active_workflow';persist();time.sleep(10)
        probe=acquire_gpu_lock();probe.close()
        import ctypes
        sleep_guard=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        from .storage_budget import check_budget
        check_budget()
        reservation=storage_reservation(jobs,registry['records'])
        atomic_write_json(directory/'storage_preflight.json',reservation)
        if reservation['additional_cache_bytes']:
            check_budget(additional_cache_bytes=reservation['additional_cache_bytes'])
        resolved=OUT/'resolved_jobs';resolved.mkdir(exist_ok=True)
        for index,job in enumerate(jobs):
            verify_release()
            record=registry['records'].setdefault(job['id'],dict(state='pending',job=job,attempts=[]))
            if record['job']!=job:raise ValueError('Scientific job identity changed')
            if record['state']=='completed':validate_completed(record);continue
            if record.get('attempts'):
                previous=record['attempts'][-1].get('run_directory')
                if previous and (Path(previous)/'summary.json').exists() and (Path(previous)/'status.json').exists():
                    status=read(Path(previous)/'status.json')['status']
                    if status in ['completed_subset_pretraining','completed_downstream_task','completed_paper_lp_test']:
                        commit(job,record,previous);persist();continue
            if pause.exists():raise TrainingPaused('Pause retained at scientific job boundary')
            if job['kind']=='reference_encoder':reference_encoder(job,record);persist();continue
            if job.get('dependency') and registry['records'].get(job['dependency'],{}).get('state')!='completed':raise ValueError('Complete matching encoder prerequisite required')
            if job.get('dependency'):validate_completed(registry['records'][job['dependency']])
            spec=copy.deepcopy(job)
            if job.get('dependency'):spec['encoder_checkpoint']=registry['records'][job['dependency']]['encoder_checkpoint']
            spec_path=resolved/(hashlib.sha256(job['id'].encode()).hexdigest()[:16]+'.json')
            if spec_path.exists() and read(spec_path)!=spec:raise ValueError('Resolved job cannot change')
            atomic_write_json(spec_path,spec)
            filename='last.pt' if job['kind']=='pretraining' else 'link_head_last.pt' if job.get('task')=='link' else 'probe_last.pt' if job.get('task')!='few_shot' else None
            parent=best_resume(record,'test_cursor.json') if job.get('task')=='link' else None
            if parent is None and job.get('task')=='link':
                for attempt in reversed(record.get('attempts',[])):
                    if attempt.get('run_directory'):
                        lock=Path(attempt['run_directory'])/'selection_lock.json'
                        if lock.exists():parent=lock;break
            if parent is None and filename:parent=best_resume(record,filename)
            arguments=['-m','came_repro.paper_task_runtime','--job',str(spec_path)]
            if parent:arguments+=['--resume',str(parent)]
            remaining=float('inf') if ledger['max_active_hours'] is None else ledger['max_active_hours']*3600-ledger['active_seconds']
            if remaining<=0:raise TrainingPaused('Cumulative explicit group budget exhausted; restart does not reset it')
            ledger['state']='running';registry['current_group']=directory.name;persist()
            logger.info('PAPER_JOB %d/%d %s; no warm-start; validation-only selection; active_hours=%.3f',index+1,len(jobs),job['id'],ledger['active_seconds']/3600)
            code,run=run_child(arguments,pause,registry,REGISTRY,record,logger,remaining)
            if code==75 or pause.exists():raise TrainingPaused('Durable training/evaluation boundary saved')
            if code or not run:raise RuntimeError('Paper job failed: %s exit=%s'%(job['id'],code))
            commit(job,record,run);persist();summarize(group,registry,jobs,full,directory)
            evict_owned_cache(job,registry,jobs);persist()
        ledger['state']='completed_full_group' if not args.pilot else 'completed_single_seed_diagnostic_slice'
        persist();summarize(group,registry,jobs,full,directory)
    except BaseException as error:
        ledger.update(state='paused' if isinstance(error,(TrainingPaused,KeyboardInterrupt)) else 'failed',error=str(error));persist();raise
    finally:
        if sleep_guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for h in handlers:h.close();logger.removeHandler(h)
        if workflow:workflow.close()
        own.close()
