"""Finite offline continuation: six validation jobs, then at most two tests.

The queue and source identities are frozen before starting. Test scores never
decide whether to add candidates, change recipes or select a seed.
"""
import copy
import datetime
import json
import logging
import math
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
PROFILE=ROOT/'configs/cloth_wn_autonomous_20261017.json'
LEDGER=ROOT/'artifacts/workflow_state/cloth_wn_autonomous_current.json'
PAUSE=ROOT/'artifacts/workflow_state/representation_coverage_pause.json'
DIRECTORY=ROOT/'artifacts/cloth_wn_autonomous_20261017'
BASE=ROOT/'artifacts/workflow_state/representation_coverage_current.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_profile(plan):
    if plan['schema']!='came_cloth_wn_bounded_offline_v1' or plan['seed']!=0:
        raise ValueError('Fixed one-seed validation protocol required')
    if plan['formal_repeats'] or not plan['offline'] or plan['maximum_new_test_evaluations_per_task']!=1:
        raise ValueError('Only offline single confirmations are authorized here')
    if plan['minimum_validation_gain_pp']!=.5 or plan['maximum_active_hours']!=18:
        raise ValueError('Declared selection threshold / finite budget changed')
    all_ids=[]
    for task,rates,cap,patiences in [('cloth',[.001,.0005,.0003],300,[10,20,20]),
                                    ('wn',[.05,.01,.005],2000,[100,100,100])]:
        candidates=plan[task+'_candidates']
        if len(candidates)!=3 or [r['lr'] for r in candidates]!=rates:
            raise ValueError('Three predetermined validation candidates required')
        if [r['patience'] for r in candidates]!=patiences or any(r['max_epochs']!=cap for r in candidates):
            raise ValueError('Declared epoch budget / patience changed')
        all_ids += [r['id'] for r in candidates]
    if len(set(all_ids))!=6:raise ValueError('Candidate IDs must be unique')


def select_validation(plan,task,summaries):
    candidates=plan[task+'_candidates'];ids=[r['id'] for r in candidates]
    if set(summaries)!=set(ids):raise ValueError('Complete all declared validation candidates before selection')
    key='validation_mrr' if task=='cloth' else 'validation_accuracy'
    for name in ids:
        item=summaries[name]
        if item.get('test_evaluations')!=0 or item.get('seed')!=0:
            raise ValueError('Candidate must be seed0 validation-only')
        if not math.isfinite(item[key]):raise ValueError('Nonfinite validation score')
    best=max(ids,key=lambda name:summaries[name][key])
    score=summaries[best][key]
    passed=score-plan['reference_validation'][task]>=plan['minimum_validation_gain_pp']/100-1e-12
    return {'candidate_id':best,'validation_score':score,
            'validation_gain_pp':100*(score-plan['reference_validation'][task]),
            'passes_validation_gate':passed,'test_used_for_selection':False,
            'all_declared_validation_jobs_completed':True}


def require_all_validation_complete(plan,records):
    for task in ('cloth','wn'):
        for candidate in plan[task+'_candidates']:
            if records.get('validate_'+candidate['id'],{}).get('state')!='completed':
                raise ValueError('All six validation jobs must finish before any new test')


def artifact_hashes(run):
    from came_repro.all_unmet_repair_pipeline import collect_protocol_artifacts
    from came_repro.author_loader import fingerprint
    hashes=collect_protocol_artifacts(run)
    location=Path(run)/'checkpoints/location.json'
    if location.exists():
        directory=Path(read(location)['checkpoint_directory'])
        for name in ('last.pt','encoder_final.pt','probe_last.pt','probe_best.pt',
                     'probe_best_raw_linear.pt','link_head_last.pt','link_head_best.pt'):
            path=directory/name
            if path.exists():hashes[str(path)]=fingerprint(path)
    return hashes


def independent_wn(run,lock_path):
    import csv
    import numpy as np
    from came_repro.author_loader import fingerprint
    from came_repro.prepared_data import prepared_directory
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from came_repro.filesystem_paths import windows_io_path
    lock=read(lock_path);selected=Path(lock['validation_run_directory']);summary=read(Path(run)/'summary.json')
    if summary['test_evaluations']!=1 or not summary['confirmation_without_retraining']:
        raise ValueError('Saved-head one-test confirmation required')
    if read(Path(run)/'manifest.json')['parent_sha256']!=lock['classifier_sha256']:
        raise ValueError('Confirmation parent SHA changed')
    scores={}
    for split,file in [('valid',selected/'validation_predictions.npz'),('test',Path(run)/'test_predictions.npz')]:
        triples=np.load(prepared_directory('WN18RR')/(split+'_triples.npy'),mmap_mode='r')
        with np.load(file) as saved:
            np.testing.assert_array_equal(saved['pairs'],triples[:,[0,2]])
            np.testing.assert_array_equal(saved['labels'],triples[:,1])
            scores[split]=float(np.mean(saved['prediction']==saved['labels']))
    if scores['test']!=summary['test_accuracy'] or scores['valid']!=summary['validation_accuracy']:
        raise ValueError('Independent official accuracy recomputation differs')
    if len(np.load(prepared_directory('WN18RR')/'test_triples.npy',mmap_mode='r'))!=3134:
        raise ValueError('Full official WN test required')
    rows=[json.loads(x) for x in (selected/'logs/metrics.jsonl').read_text().splitlines()]
    with (selected/'logs/metrics.csv').open(newline='') as f:csv_rows=list(csv.DictReader(f))
    if len(rows)!=len(csv_rows) or len(rows)!=summary['epochs_executed']:
        raise ValueError('Full CSV/JSONL epoch history differs')
    for row,other in zip(rows,csv_rows):
        for k,v in row.items():
            if float(other[k])!=float(v):raise ValueError('CSV metric differs')
    events=EventAccumulator(str(windows_io_path(selected/'tensorboard')),size_guidance={'scalars':0});events.Reload()
    if len(events.Scalars('val_accuracy'))!=len(rows):raise ValueError('WN TensorBoard history incomplete')
    for event,row in zip(events.Scalars('val_accuracy'),rows):
        if event.step!=row['step'] or not np.isclose(event.value,row['val_accuracy'],rtol=1e-6,atol=1e-8):
            raise ValueError('WN TensorBoard differs')
    for path,expected in [(selected/'summary.json',lock['validation_summary_sha256']),
                           (Path(lock['classifier_checkpoint']),lock['classifier_sha256'])]:
        if fingerprint(path)!=expected:raise ValueError('Selection artifact changed')
    value=scores['test']*100
    return {'test_accuracy_pct':value,'paper_accuracy_pct':81.97,'gap_pp':value-81.97,
            'within_two_pp':abs(value-81.97)<=2,'full_official_predictions_verified':True,
            'CSV_JSONL_TensorBoard_verified':True,'figure_visual_verified':False}


def main():
    from came_repro.author_loader import fingerprint,verify_originals
    from came_repro.atomic_io import atomic_write_json
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.storage_budget import check_budget
    from came_repro.pause_control import TrainingPaused
    plan=read(PROFILE);validate_profile(plan);verify_originals()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    base=read(BASE);sources=dict(base['identity']['sources'])
    for name in ('scripts/cloth_wn_autonomous.py','scripts/cloth_wn_autonomous_backend.py',
                 'scripts/cloth_wn_repair_entry.py','scripts/run_08_cloth_wn_repair.py'):
        path=ROOT/name;sources[str(path)]=fingerprint(path)
    identity={'sources':sources,'autonomous_profile_sha256':fingerprint(PROFILE),
              'base_identity':base['identity'],'source_ledger_sha256_at_adoption':fingerprint(BASE)}
    guard=acquire_gpu_lock(ROOT/'.runtime/representation_coverage.lock')
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    ledger=read(LEDGER) if LEDGER.exists() else {'identity':identity,'state':'prepared','records':{},
        'active_seconds':0.,'selection':{},'results':{},'rejections':{},'formal_repeats_started':0}
    if ledger['identity']!=identity:
        guard.close();raise ValueError('Frozen autonomous identity changed; refuse continuation')
    if 'fresh_pretraining' not in ledger['records']:
        ledger['records']['fresh_pretraining']=copy.deepcopy(base['records']['fresh_pretraining'])
        ledger['records']['fresh_pretraining']['state']='ready_to_resume'
    accepted_path=DIRECTORY/'acceptance.json'
    if 'new_backend_CUDA_resume' not in ledger['records'] and accepted_path.exists():
        accepted=read(accepted_path)
        if (accepted.get('passed') and accepted.get('actual_Torch_and_DGL_CUDA')
                and accepted.get('backend_sha256')==fingerprint(ROOT/'scripts/cloth_wn_autonomous_backend.py')
                and accepted.get('profile_sha256')==fingerprint(PROFILE)):
            ledger['records']['new_backend_CUDA_resume']={'state':'completed','attempts':[],
                'summary':accepted,'artifact_sha256':{str(accepted_path):fingerprint(accepted_path)}}
    atomic_write_json(LEDGER,ledger)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logger=logging.getLogger('cloth_wn_autonomous');logger.setLevel(logging.INFO)
    handlers=[logging.StreamHandler(),logging.FileHandler(DIRECTORY/('pipeline_'+stamp+'.log'),encoding='utf-8')]
    for handler in handlers:
        formatter=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s')
        import time
        formatter.converter=time.gmtime;handler.setFormatter(formatter);logger.addHandler(handler)
    sleep_guard=False
    if os.name=='nt':
        import ctypes
        sleep_guard=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    def persist():atomic_write_json(LEDGER,ledger)
    def verify():
        if PAUSE.exists():raise TrainingPaused('User/budget pause retained')
        verify_originals()
        if fingerprint(PROFILE)!=identity['autonomous_profile_sha256'] or fingerprint(BASE)!=identity['source_ledger_sha256_at_adoption']:
            raise ValueError('Frozen plan / historical source ledger changed')
        for path,expected in [(ROOT/'configs/representation_coverage_validation_20261016.json',base['identity']['plan_sha256']),
                              (ROOT/'configs/representation_coverage_20261016.json',base['identity']['config_sha256'])]:
            if fingerprint(path)!=expected:raise ValueError('Original frozen pretraining / inference plan changed')
        for path,digest in sources.items():
            if fingerprint(path)!=digest:raise ValueError('Frozen source changed: '+path)
    def execute(job,arguments,resume_name=None,resume_flag='--resume-head',accepted=None):
        verify();record=ledger['records'].setdefault(job,{'state':'ready','attempts':[]})
        if record['state']=='completed':
            for path,digest in record['artifact_sha256'].items():
                if fingerprint(path)!=digest:raise ValueError('Completed artifact changed')
            return record
        if job.startswith('confirm_'):
            for attempt in record.get('attempts',[]):
                run=attempt.get('run_directory')
                if run and any((Path(run)/name).exists() for name in ('test_predictions.npz','test_link_ranks.npz')):
                    raise ValueError('Existing interrupted test output: do not retest automatically')
        parent=best_resume(record,resume_name) if resume_name else None
        if parent:arguments += [resume_flag,str(parent)]
        remaining=plan['maximum_active_hours']*3600-ledger['active_seconds']
        if remaining<=0:raise TrainingPaused('Bounded active-runtime budget exhausted')
        ledger['state']='running_'+job;persist();logger.info('AUTO_STAGE %s parent=%s',job,parent)
        code,run=run_child(arguments,PAUSE,ledger,LEDGER,record,logger,remaining)
        if code==75 or PAUSE.exists():raise TrainingPaused('Saved current stage at cooperative boundary')
        if code:raise RuntimeError(f'GPU child {job} exited {code}; stop rather than mask a data/runtime error')
        if accepted:
            summary=read(accepted)
            if not summary['passed'] or summary['backend_sha256']!=fingerprint(ROOT/'scripts/cloth_wn_autonomous_backend.py') or summary['profile_sha256']!=fingerprint(PROFILE):
                raise ValueError('New exact CUDA/resume acceptance failed')
            record.update(state='completed',summary=summary,artifact_sha256={str(accepted):fingerprint(accepted)})
        else:
            if not run:raise ValueError('Child produced no unique run')
            record.update(state='completed',run_directory=run,summary=read(Path(run)/'summary.json'),artifact_sha256=artifact_hashes(run))
        persist();return record
    try:
        verify();probe=acquire_gpu_lock();probe.close()
        check_budget(additional_project_bytes=512*2**20,additional_cache_bytes=4*2**30)
        logger.info('OFFLINE finite queue: resume 20480 updates; 3 validation candidates per task; test once per selected task; no formal repeats.')
        backend=ROOT/'scripts/cloth_wn_autonomous_backend.py'
        execute('new_backend_CUDA_resume',[str(backend),'--stage','acceptance'],accepted=DIRECTORY/'acceptance.json')
        old_backend=ROOT/'scripts/representation_coverage_backend.py'
        trained=execute('fresh_pretraining',[str(old_backend),'--stage','pretrain'],'last.pt','--resume')
        summary=trained['summary']
        if summary.get('step')!=20480 or summary.get('completed_subset_rounds')!=8 or not summary.get('complete_declared_subset_pretraining'):
            raise ValueError('Finish exact original declared pretraining budget first')
        run=Path(trained['run_directory']);encoder=Path(read(run/'checkpoints/location.json')['checkpoint_directory'])/'encoder_final.pt'
        encoder_hash=fingerprint(encoder)
        # No test is run until all six predeclared validation jobs are complete.
        for task in ('cloth','wn'):
            execute('encode_'+task,[str(old_backend),'--stage','encode_'+task,'--checkpoint',str(encoder)])
            summaries={}
            for candidate in plan[task+'_candidates']:
                name=candidate['id']
                record=execute('validate_'+name,[str(backend),'--stage','probe','--candidate',name,'--checkpoint',str(encoder)],
                               'link_head_last.pt' if task=='cloth' else 'probe_last.pt')
                summaries[name]=record['summary']
                if summaries[name]['encoder_sha256']!=encoder_hash:raise ValueError('Candidate encoder changed')
            ledger['selection'][task]=select_validation(plan,task,summaries);persist()
        require_all_validation_complete(plan,ledger['records'])
        for task in ('cloth','wn'):
            selected=ledger['selection'][task]
            if not selected['passes_validation_gate']:
                ledger['rejections'][task]=dict(selected,new_test_evaluations=0,previous_test_retained=True);persist();continue
            record=ledger['records']['validate_'+selected['candidate_id']]
            run=Path(record['run_directory']);summary=record['summary']
            config=read(run/'config.json')
            from priority_refinement_backend import reference_cache
            cache=reference_cache(summary,config)
            directory=Path(read(run/'checkpoints/location.json')['checkpoint_directory'])
            head=directory/('link_head_best.pt' if task=='cloth' else 'probe_best_raw_linear.pt')
            lock={'dataset':summary['dataset'],'encoder_sha256':encoder_hash,
                'classifier_checkpoint':str(head),'classifier_sha256':fingerprint(head),
                'validation_run_directory':str(run),'validation_summary_sha256':fingerprint(run/'summary.json'),
                'embedding_cache':str(cache),'inference_state_sha256':fingerprint(cache/'inference_state.json'),
                'locked_before_test':True,'test_used_for_selection':False,'graph_policy':summary['graph_policy'],
                'all_six_validation_jobs_completed_before_any_test':True,'selected_candidate':selected['candidate_id']}
            if task=='cloth':lock.update(head_variant=summary['head_variant'],selected_validation_mrr=summary['validation_mrr'])
            else:lock.update(feature_transform='train_pair_affine',selected_validation_accuracy=summary['validation_accuracy'])
            lock_path=DIRECTORY/(task+'_selection_lock.json')
            if lock_path.exists() and read(lock_path)!=lock:raise ValueError('Frozen selection lock changed')
            atomic_write_json(lock_path,lock)
            args=(['-m','came_repro.protocol_repair','--stage','confirm_lp','--selection-lock',str(lock_path)] if task=='cloth'
                  else ['-m','came_repro.kg_selected_confirmation','--selection-lock',str(lock_path)])
            confirmed=execute('confirm_'+task,args)
            if task=='cloth':
                from audit_two_lp_confirmation import audit_confirmation
                result=audit_confirmation(Path(confirmed['run_directory']),lock_path)
            else:result=independent_wn(Path(confirmed['run_directory']),lock_path)
            result['manual_figure_visual_review_pending']=True
            ledger['results'][task]=result;persist()
        ledger.update(state='completed_bounded_offline_search_pending_visual_review',
            all_targets_met=all(ledger['results'].get(t,{}).get('within_two_pp',False) for t in ('cloth','wn')))
        persist()
        lines=['# Cloth / WN 自动诊断结果','', '本轮为固定候选单seed诊断，非正式十seed完成。', '']
        for task in ('cloth','wn'):
            lines += ['## '+task,'',json.dumps(ledger['results'].get(task,ledger['rejections'].get(task)),ensure_ascii=False,indent=2),'']
        (DIRECTORY/'结果摘要.md').write_text('\n'.join(lines),encoding='utf-8')
        logger.info('AUTO_COMPLETED all_targets_met=%s; no more candidates will be added using test scores.',ledger['all_targets_met'])
    except BaseException as error:
        ledger.update(state='paused' if isinstance(error,TrainingPaused) or PAUSE.exists() else 'failed',error=str(error));persist();raise
    finally:
        if sleep_guard:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for handler in handlers:handler.close();logger.removeHandler(handler)
        guard.close()


if __name__=='__main__':main()
