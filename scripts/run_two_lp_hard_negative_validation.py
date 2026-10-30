"""Finite train-only hard-negative diagnostics after the current two LP trials exit."""
import datetime
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file,atomic_write_json
    from came_repro.author_loader import fingerprint,verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.all_unmet_repair_pipeline import collect_protocol_artifacts
    from came_repro.storage_budget import check_budget
    profile_path=ROOT/'configs/lp_negative_objective_20261014.json';profile=read_json_file(profile_path)
    sources={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}
    script=ROOT/'scripts/run_lp_hard_negative_candidate.py';identity={'profile_sha256':fingerprint(profile_path),'candidate_script_sha256':fingerprint(script),'self_sha256':fingerprint(__file__),'sources':sources}
    path=ROOT/'artifacts/workflow_state/two_lp_hard_negative_current.json';pause=ROOT/'artifacts/workflow_state/two_lp_hard_negative_pause.json'
    ledger=read_json_file(path) if path.exists() else {'identity':identity,'state':'waiting_for_two_lp_convergence_completion','records':{},'active_seconds':0.,'ready_tasks':[],'verified_reports':{}}
    if ledger['identity']!=identity:raise ValueError('Hard-negative identity changed')
    guard=acquire_gpu_lock(ROOT/'.runtime/two_lp_hard_negative.lock');atomic_write_json(path,ledger)
    try:
        while True:
            if pause.exists():ledger['state']='paused';atomic_write_json(path,ledger);return
            first=read_json_file(ROOT/'artifacts/workflow_state/two_lp_priority_current.json')
            if first['state'] in ('failed','paused'):
                ledger['state']='deferred_previous_failure_or_pause';atomic_write_json(path,ledger);return
            if first['state']=='completed_two_lp_bounded_diagnostics':
                try:gpu=acquire_gpu_lock();gpu.close();break
                except RuntimeError:pass
            time.sleep(15)
        verify_originals();check_budget(additional_cache_bytes=profile['budget']['checkpoint_metadata_reservation_bytes'])
        original=read_json_file(ROOT/'artifacts/workflow_state/all_unmet_repair_current.json')
        proof_time=profile_path.stat().st_mtime
        first_test_time=datetime.datetime.fromisoformat(first['records']['confirm_sports-copurchase']['attempts'][0]['started_utc']).timestamp()
        if proof_time>=first_test_time:raise ValueError('This candidate was not declared before the convergence test')
        remaining=[d for d in profile['tasks'] if d not in first['ready_tasks']]
        ledger['protected_newly_ready_tasks']=first['ready_tasks'];ledger['declaration_before_first_convergence_test_verified']=True;atomic_write_json(path,ledger)
        if not remaining:ledger['state']='completed_no_failed_tasks_to_repair';atomic_write_json(path,ledger);return
        directory=ROOT/'artifacts/remaining_main_pilots/hard_lp';directory.mkdir(parents=True,exist_ok=True)
        logger=logging.getLogger('hard_lp');logger.setLevel(logging.INFO)
        for handler in [logging.StreamHandler(),logging.FileHandler(directory/('pipeline_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log'),encoding='utf-8')]:handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logger.addHandler(handler)
        def verify_identity():
            for name,digest in sources.items():
                if fingerprint(ROOT/name)!=digest:raise ValueError('Frozen hard LP source changed')
            if fingerprint(script)!=identity['candidate_script_sha256'] or fingerprint(__file__)!=identity['self_sha256'] or fingerprint(profile_path)!=identity['profile_sha256']:raise ValueError('Frozen hard LP profile/script changed')
        verify_identity()
        acceptance_path=ROOT/'artifacts/audit/hard_lp_CUDA_acceptance_20261014.json'
        if not ledger.get('CUDA_acceptance_passed'):
            code=subprocess.run([sys.executable,'-u','-m','came_repro.suite_job','--',str(script),'--acceptance'],cwd=ROOT,env={**os.environ,'CAME_PAUSE_FILE':str(pause),'PYTHONPATH':str(ROOT/'src'),'PYTHONIOENCODING':'utf-8'}).returncode
            if code or not read_json_file(acceptance_path)['passed']:raise RuntimeError('Hard LP actual CUDA/resume acceptance failed')
            ledger['CUDA_acceptance_passed']=True;ledger['acceptance_sha256']=fingerprint(acceptance_path);atomic_write_json(path,ledger)
        def execute(jid,args,resume_name=None):
            verify_identity();rec=ledger['records'].setdefault(jid,{'state':'ready','attempts':[]})
            if rec['state']=='completed':
                for name,digest in rec['artifact_sha256'].items():
                    if fingerprint(name)!=digest:raise ValueError('Completed hard LP artifacts changed')
                return rec
            if resume_name:
                cp=best_resume(rec,resume_name)
                if cp:args+=['--resume-head',str(cp)]
            ledger['state']='running';code,run=run_child(args,pause,ledger,path,rec,logger,float('inf'))
            if code or not run:
                ledger['state']=rec['state']='paused' if code==75 else 'failed';atomic_write_json(path,ledger);raise RuntimeError('Hard LP child stopped '+jid)
            rec.update(state='completed',run_directory=run,summary=read_json_file(Path(run)/'summary.json'),artifact_sha256=collect_protocol_artifacts(run));atomic_write_json(path,ledger);return rec
        for dataset in remaining:
            options=[original['records']['validate_'+dataset],first['records']['validate_'+dataset]]
            reference=max(options,key=lambda r:r['summary']['validation_mrr'])
            expected={'run_directory':reference['run_directory'],'summary_sha256':fingerprint(Path(reference['run_directory'])/'summary.json'),'validation_mrr':reference['summary']['validation_mrr']}
            if dataset in ledger.get('references',{}) and ledger['references'][dataset]!=expected:raise ValueError('Hard LP validation reference changed')
            ledger.setdefault('references',{})[dataset]=expected;atomic_write_json(path,ledger)
            candidate=execute('validate_'+dataset,[str(script),'--dataset',dataset,'--reference-run',reference['run_directory']],'link_head_last.pt')
            gain=100*(candidate['summary']['validation_mrr']-reference['summary']['validation_mrr'])
            if gain<profile['gate']['minimum_validation_gain_pp']:
                ledger.setdefault('rejections',{})[dataset]={'validation_gain_pp':gain,'new_test_evaluations':0};atomic_write_json(path,ledger);continue
            run=Path(candidate['run_directory']);s=candidate['summary'];cache=Path(s['embedding_cache']);best=Path(read_json_file(run/'checkpoints/location.json')['checkpoint_directory'])/'link_head_best.pt'
            lock={'dataset':dataset,'encoder_sha256':s['encoder_sha256'],'classifier_checkpoint':str(best),'classifier_sha256':fingerprint(best),'validation_run_directory':str(run),'validation_summary_sha256':fingerprint(run/'summary.json'),'embedding_cache':str(cache),'inference_state_sha256':fingerprint(cache/'inference_state.json'),'locked_before_test':True,'test_used_for_selection':False,'selected_validation_mrr':s['validation_mrr'],'head_variant':s['head_variant'],'graph_policy':'upstream_train_bidirected_unique'}
            lock_path=directory/(dataset+'_selection_lock.json')
            if lock_path.exists() and read_json_file(lock_path)!=lock:raise ValueError('Hard LP selection lock changed')
            atomic_write_json(lock_path,lock)
            confirmed=execute('confirm_'+dataset,['-m','came_repro.protocol_repair','--stage','confirm_lp','--selection-lock',str(lock_path)])
            final=Path(confirmed['run_directory']);pred=np.load(final/'test_link_ranks.npz');raw=ROOT/'data/raw/mm_graph'/dataset/'lp-edge-split.pt';split=torch.load(str(raw),map_location='cpu',weights_only=False,mmap=True)
            for key in ['source_node','target_node']:
                if not np.array_equal(pred[key],np.asarray(split['test'][key])):raise ValueError('Hard LP official query IDs mismatch')
            n=50250 if dataset=='sports-copurchase' else 125839;train_keys=np.unique(np.asarray(split['train']['source_node'],dtype=np.int64)*n+np.asarray(split['train']['target_node'],dtype=np.int64));reverse=np.isin(pred['target_node'].astype(np.int64)*n+pred['source_node'],train_keys)
            if not np.array_equal(pred['reverse_overlap'],reverse):raise ValueError('Hard LP reverse overlap mismatch')
            ranks=pred['rank'].astype(np.float64);mrr=float((1/ranks).mean());paper=37.68 if dataset=='sports-copurchase' else 28.88
            if not np.isfinite(ranks).all() or (ranks<1).any() or (ranks>151).any() or abs(mrr-confirmed['summary']['test']['official']['mrr'])>2e-7:raise ValueError('Independent hard LP official MRR mismatch')
            report={'dataset':dataset,'run_directory':str(final),'official_queries':len(ranks),'official_negative_candidates':int(np.asarray(split['test']['target_node_neg']).shape[1]),'raw_split_sha256':fingerprint(raw),'test_mrr_pct':100*mrr,'paper_mrr_pct':paper,'gap_pp':100*mrr-paper,'within_two_pp':abs(100*mrr-paper)<=2,'validation_gain_pp':gain,'reverse_overlap_queries':int(reverse.sum()),'nonoverlap_mrr_pct':float((1/ranks[~reverse]).mean())*100,'known_train_only_hard_candidate':True,'source_target_exact':True,'test_evaluations':1,'PNG_visual_verified':False,'formal_repeats_not_started':True}
            atomic_write_json(ROOT/'artifacts/audit'/('hard_'+dataset+'_verified_20261014.json'),report);ledger['verified_reports'][dataset]=report
            if report['within_two_pp'] and dataset not in ledger['ready_tasks']:ledger['ready_tasks'].append(dataset)
            atomic_write_json(path,ledger)
        ledger['state']='completed_bounded_hard_negative_diagnostics';atomic_write_json(path,ledger)
    except BaseException as error:
        if ledger['state']!='paused':ledger.update(state='failed',error=str(error));atomic_write_json(path,ledger)
        raise
    finally:guard.close()


if __name__=='__main__':main()
