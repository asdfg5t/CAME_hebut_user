"""User-priority Sports/Cloth bounded CUDA trials; deliver only independently verified scores."""
import datetime
import logging
import math
import sys
import time
from pathlib import Path

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def verify_test_record(dataset,record,reference,paper):
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file
    from came_repro.author_loader import fingerprint
    run=Path(record['run_directory']);s=record['summary'];p=np.load(run/'test_link_ranks.npz')
    raw=ROOT/'data/raw/mm_graph'/dataset/'lp-edge-split.pt';split=torch.load(str(raw),map_location='cpu',weights_only=False,mmap=True)
    for key in ['source_node','target_node']:
        if not np.array_equal(p[key],np.asarray(split['test'][key])):raise ValueError('Saved LP query IDs not complete official test')
    n=50250 if dataset=='sports-copurchase' else 125839
    keys=np.unique(np.asarray(split['train']['source_node'],dtype=np.int64)*n+np.asarray(split['train']['target_node'],dtype=np.int64))
    reverse=np.isin(np.asarray(split['test']['target_node'],dtype=np.int64)*n+np.asarray(split['test']['source_node'],dtype=np.int64),keys)
    if not np.array_equal(reverse,p['reverse_overlap']):raise ValueError('Reverse-overlap audit changed')
    rank=p['rank'].astype(np.float64)
    if not np.isfinite(rank).all() or (rank<1).any() or (rank>151).any():raise ValueError('Official full rank range invalid')
    mrr=float((1/rank).mean());clean=float((1/rank[~reverse]).mean())
    if abs(mrr-s['test']['official']['mrr'])>2e-7 or abs(clean-s['test']['reverse_nonoverlap_subset']['mrr'])>2e-7:raise ValueError('Independent official MRR mismatch')
    selected=Path(s['validation_run_directory']);lock_path=ROOT/'artifacts/remaining_main_pilots/two_lp_priority'/(dataset+'_selection_lock.json');lock=read_json_file(lock_path)
    if not lock['locked_before_test'] or lock['test_used_for_selection'] or fingerprint(lock['classifier_checkpoint'])!=lock['classifier_sha256'] or fingerprint(selected/'summary.json')!=lock['validation_summary_sha256'] or fingerprint(lock_path)!=s['selection_lock_sha256']:
        raise ValueError('LP confirmation saved-weight lock mismatch')
    if read_json_file(run/'manifest.json')['parent_sha256']!=lock['classifier_sha256']:raise ValueError('LP selected head parent SHA mismatch')
    if not list(selected.rglob('events.out.tfevents.*')) or not (selected/'figures/protocol_lp_curves.png').exists():raise ValueError('Required LP training figure/events missing')
    report={'dataset':dataset,'run_directory':str(run),'official_queries':len(rank),'official_negative_candidates':int(np.asarray(split['test']['target_node_neg']).shape[1]),'raw_split_sha256':fingerprint(raw),'source_target_exact':True,'test_mrr_pct':mrr*100,'paper_mrr_pct':paper,'gap_pp':mrr*100-paper,'within_two_pp':abs(mrr*100-paper)<=2,'reverse_overlap_queries':int(reverse.sum()),'nonoverlap_mrr_pct':clean*100,'validation_gain_pp':100*(s['validation_mrr']-reference['summary']['validation_mrr']),'locked_saved_head_and_parent_SHA_verified':True,'test_evaluations':1,'PNG_visual_verified':False,'formal_repeats_not_started':True}
    return report


def main():
    from came_repro.atomic_io import read_json_file,atomic_write_json
    from came_repro.author_loader import fingerprint,verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.all_unmet_repair_pipeline import collect_protocol_artifacts
    from came_repro.storage_budget import check_budget
    profile_path=ROOT/'configs/two_lp_priority_20261014.json';profile=read_json_file(profile_path)
    candidate_path=ROOT/profile['candidate_profile'];sources={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}
    identity={'sources':sources,'self_sha256':fingerprint(__file__),'candidate_script_sha256':fingerprint(ROOT/'scripts/run_convergence_candidate.py'),'candidate_profile_sha256':fingerprint(candidate_path),'priority_profile_sha256':fingerprint(profile_path)}
    acceptance=read_json_file(ROOT/'artifacts/audit/convergence_cuda_acceptance_20261014.json')
    if not acceptance['passed'] or acceptance['profile_sha256']!=identity['candidate_profile_sha256'] or acceptance['script_sha256']!=identity['candidate_script_sha256']:raise ValueError('Frozen CUDA/resume acceptance required')
    path=ROOT/'artifacts/workflow_state/two_lp_priority_current.json';pause=ROOT/'artifacts/workflow_state/two_lp_priority_pause.json'
    ledger=read_json_file(path) if path.exists() else {'identity':identity,'state':'waiting_for_WN_safe_pause','records':{},'active_seconds':0.,'verified_reports':{},'ready_tasks':[]}
    if ledger['identity']!=identity:raise ValueError('Priority LP identity changed')
    guard=acquire_gpu_lock(ROOT/'.runtime/two_lp_priority.lock');atomic_write_json(path,ledger)
    try:
        while True:
            if pause.exists():ledger['state']='paused';atomic_write_json(path,ledger);return
            old=read_json_file(ROOT/'artifacts/workflow_state/convergence_repair_current.json')
            if old['state']=='paused':
                try:gpu=acquire_gpu_lock();gpu.close();break
                except RuntimeError:pass
            elif old['state']=='failed':raise ValueError('Prior convergence failed: audit before priority launch')
            time.sleep(5)
        verify_originals();check_budget(additional_cache_bytes=profile['resources']['checkpoint_metadata_reservation_bytes'])
        reference_ledger=read_json_file(ROOT/profile['reference_ledger']);references={dataset:reference_ledger['records']['validate_'+dataset] for dataset in profile['tasks']}
        expected={dataset:{'run_directory':r['run_directory'],'summary_sha256':fingerprint(Path(r['run_directory'])/'summary.json')} for dataset,r in references.items()}
        if 'references' in ledger and ledger['references']!=expected:raise ValueError('LP reference changed')
        ledger['references']=expected;atomic_write_json(path,ledger)
        directory=ROOT/'artifacts/remaining_main_pilots/two_lp_priority';directory.mkdir(parents=True,exist_ok=True)
        logger=logging.getLogger('two_lp_priority');logger.setLevel(logging.INFO)
        for handler in [logging.StreamHandler(),logging.FileHandler(directory/('pipeline_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log'),encoding='utf-8')]:
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logger.addHandler(handler)
        def execute(jid,args,resume_name=None):
            for name,digest in sources.items():
                if fingerprint(ROOT/name)!=digest:raise ValueError('Frozen LP source changed')
            if fingerprint(candidate_path)!=identity['candidate_profile_sha256'] or fingerprint(ROOT/'scripts/run_convergence_candidate.py')!=identity['candidate_script_sha256'] or fingerprint(__file__)!=identity['self_sha256'] or fingerprint(profile_path)!=identity['priority_profile_sha256']:
                raise ValueError('Frozen LP script/profile changed')
            rec=ledger['records'].setdefault(jid,{'state':'ready','attempts':[]})
            if rec['state']=='completed':
                for name,digest in rec['artifact_sha256'].items():
                    if fingerprint(name)!=digest:raise ValueError('Completed LP artifacts changed')
                return rec
            if resume_name:
                cp=best_resume(rec,resume_name)
                if cp:args+=['--resume-head',str(cp)]
            ledger['state']='running';code,run=run_child(args,pause,ledger,path,rec,logger,float('inf'))
            if code or not run:
                ledger['state']=rec['state']='paused' if code==75 else 'failed';atomic_write_json(path,ledger);raise RuntimeError('Priority LP child stopped '+jid)
            rec.update(state='completed',run_directory=run,summary=read_json_file(Path(run)/'summary.json'),artifact_sha256=collect_protocol_artifacts(run));atomic_write_json(path,ledger);return rec
        for ordinal,dataset in enumerate(profile['tasks'],1):
            logger.info('PRIORITY_VALIDATION %d/2 dataset=%s; only full validation chooses head',ordinal,dataset)
            candidate=execute('validate_'+dataset,['scripts/run_convergence_candidate.py','--stage','lp','--dataset',dataset,'--reference-run',references[dataset]['run_directory']],'link_head_last.pt')
            gain=100*(candidate['summary']['validation_mrr']-references[dataset]['summary']['validation_mrr'])
            if not math.isfinite(gain):raise ValueError('Nonfinite LP validation gain')
            if gain<profile['minimum_validation_gain_pp']:
                ledger.setdefault('rejections',{})[dataset]={'validation_gain_pp':gain,'new_test_evaluations':0};atomic_write_json(path,ledger);continue
            run=Path(candidate['run_directory']);s=candidate['summary'];cache=Path(s['embedding_cache']);best=Path(read_json_file(run/'checkpoints/location.json')['checkpoint_directory'])/'link_head_best.pt'
            selected={'dataset':dataset,'encoder_sha256':s['encoder_sha256'],'classifier_checkpoint':str(best),'classifier_sha256':fingerprint(best),'validation_run_directory':str(run),'validation_summary_sha256':fingerprint(run/'summary.json'),'embedding_cache':str(cache),'inference_state_sha256':fingerprint(cache/'inference_state.json'),'locked_before_test':True,'test_used_for_selection':False,'selected_validation_mrr':s['validation_mrr'],'head_variant':s['head_variant'],'graph_policy':'upstream_train_bidirected_unique'}
            lock_path=directory/(dataset+'_selection_lock.json')
            if lock_path.exists() and read_json_file(lock_path)!=selected:raise ValueError('LP saved-weight lock changed')
            atomic_write_json(lock_path,selected)
            confirmed=execute('confirm_'+dataset,['-m','came_repro.protocol_repair','--stage','confirm_lp','--selection-lock',str(lock_path)])
            report=verify_test_record(dataset,confirmed,references[dataset],profile['paper_mrr_pct'][dataset])
            report_path=ROOT/'artifacts/audit'/('priority_'+dataset+'_verified_20261014.json');atomic_write_json(report_path,report)
            ledger['verified_reports'][dataset]=report
            if report['within_two_pp'] and dataset not in ledger['ready_tasks']:ledger['ready_tasks'].append(dataset)
            atomic_write_json(path,ledger)
        ledger['state']='completed_two_lp_bounded_diagnostics';atomic_write_json(path,ledger)
        atomic_write_json(directory/'delivery_eligibility.json',{'ready_tasks':ledger['ready_tasks'],'reports':ledger['verified_reports'],'eligibility_requires_visual_and_full_repeat_entry_acceptance':True,'formal_training_started_by_agent':False,'all_unmet_fixed':len(ledger['ready_tasks'])==2})
    except BaseException as error:
        if ledger['state']!='paused':ledger.update(state='failed',error=str(error));atomic_write_json(path,ledger)
        raise
    finally:guard.close()


if __name__=='__main__':main()
