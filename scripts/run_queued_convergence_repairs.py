"""Serialize accepted bounded convergence trials after Products capacity calibration."""
import copy
import datetime
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

if __name__=='__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from came_repro.atomic_io import read_json_file,atomic_write_json
    from came_repro.author_loader import fingerprint,verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.all_unmet_repair_pipeline import collect_protocol_artifacts
    from came_repro.storage_budget import check_budget
    profile_path=ROOT/'configs/convergence_repair_20261014.json'
    profile=read_json_file(profile_path)
    sources={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}
    scripts={name:fingerprint(ROOT/name) for name in ['scripts/run_convergence_candidate.py','scripts/run_queued_convergence_repairs.py']}
    identity={'profile_sha256':fingerprint(profile_path),'sources':sources,'scripts':scripts}
    path=ROOT/'artifacts/workflow_state/convergence_repair_current.json'
    pause=ROOT/'artifacts/workflow_state/convergence_repair_pause.json'
    ledger=read_json_file(path) if path.exists() else {'identity':identity,'state':'waiting_for_products_calibration','records':{},'active_seconds':0.,'test_used_for_selection':False}
    if ledger['identity']!=identity:raise ValueError('Convergence identity changed; audit before new branch')
    supervisor_lock=acquire_gpu_lock(ROOT/'.runtime/convergence_repair.lock')
    atomic_write_json(path,ledger)
    try:
        while True:
            if pause.exists():ledger['state']='paused';atomic_write_json(path,ledger);return
            prior=read_json_file(ROOT/'artifacts/workflow_state/all_unmet_repair_current.json')
            capacity=read_json_file(ROOT/'artifacts/workflow_state/products_capacity_current.json')
            if prior['state'] in ('paused','failed'):
                ledger['state']='deferred_prior_pause_or_failure';atomic_write_json(path,ledger);return
            if prior['state']=='completed_bounded_graph_text_screen' and capacity['state'] not in ('waiting_for_protocol_repair_GPU_release','calibrating'):
                try:
                    lock=acquire_gpu_lock();lock.close();break
                except RuntimeError:pass
            time.sleep(15)
        verify_originals();check_budget(additional_cache_bytes=128*2**20)
        old=read_json_file(ROOT/'artifacts/workflow_state/kg_affine_screen_current.json')
        wn_candidates=[prior['records']['validate_WN18RR']]
        # Old KG validation-selected classifier has a bound validation run, independent of test.
        # Previous affine main ledger explicitly contains the candidate record.
        for record in old['records'].values():
            if (record.get('summary') or {}).get('dataset')=='WN18RR' and record['summary'].get('test_evaluations',0)==0 and record['summary'].get('feature_transform')=='train_pair_affine':
                wn_candidates.append(record)
        wn_reference=max(wn_candidates,key=lambda r:r['summary']['validation_accuracy'])
        references={dataset:prior['records']['validate_'+dataset] for dataset in ['sports-copurchase','cloth-copurchase']}
        references['WN18RR']=wn_reference
        expected={k:{'run_directory':v['run_directory'],'validation_summary_sha256':fingerprint(Path(v['run_directory'])/'summary.json'),'validation_metric':v['summary']['validation_accuracy' if k=='WN18RR' else 'validation_mrr']} for k,v in references.items()}
        if 'references' in ledger and ledger['references']!=expected:raise ValueError('Frozen validation reference changed')
        ledger['references']=expected;atomic_write_json(path,ledger)
        logger=logging.getLogger('convergence_repair');logger.setLevel(logging.INFO)
        logdir=ROOT/'artifacts/remaining_main_pilots/convergence_repair';logdir.mkdir(parents=True,exist_ok=True)
        for handler in [logging.StreamHandler(),logging.FileHandler(logdir/('pipeline_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log'),encoding='utf-8')]:
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logger.addHandler(handler)
        def verify_identity():
            for name,digest in dict(sources,**scripts).items():
                if fingerprint(ROOT/name)!=digest:raise ValueError('Frozen convergence code changed')
            if fingerprint(profile_path)!=identity['profile_sha256']:raise ValueError('Frozen convergence profile changed')
        verify_identity()
        if not ledger.get('CUDA_acceptance_passed'):
            audit_path=ROOT/'artifacts/audit/convergence_cuda_acceptance_20261014.json'
            audit=read_json_file(audit_path) if audit_path.exists() else {}
            if not (audit.get('passed') and audit.get('profile_sha256')==identity['profile_sha256'] and audit.get('script_sha256')==scripts['scripts/run_convergence_candidate.py']):
                code=subprocess.run([sys.executable,'-u','-m','came_repro.suite_job','--','scripts/run_convergence_candidate.py','--stage','acceptance'],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'CAME_PAUSE_FILE':str(pause)}).returncode
                if code:raise RuntimeError('Convergence CUDA acceptance failed; no benchmark dispatched')
                audit=read_json_file(audit_path)
            if not audit['passed']:raise ValueError('CUDA acceptance did not pass')
            ledger['CUDA_acceptance_passed']=True;ledger['acceptance_sha256']=fingerprint(ROOT/'artifacts/audit/convergence_cuda_acceptance_20261014.json');atomic_write_json(path,ledger)
        def execute(jid,args,resume_name=None):
            verify_identity()
            if pause.exists():raise RuntimeError('Convergence pause requested')
            record=ledger['records'].setdefault(jid,{'state':'ready','attempts':[]})
            if record['state']=='completed':
                for name,digest in record['artifact_sha256'].items():
                    if fingerprint(name)!=digest:raise ValueError('Completed convergence artifact changed')
                return record
            if resume_name:
                resumed=best_resume(record,resume_name)
                if resumed:args+=['--resume-head',str(resumed)]
            ledger['state']='running'
            code,run=run_child(args,pause,ledger,path,record,logger,float('inf'))
            if code or not run:
                ledger['state']=record['state']='paused' if code==75 else 'failed';atomic_write_json(path,ledger);raise RuntimeError('Convergence child failed '+jid)
            record.update(state='completed',run_directory=run,summary=read_json_file(Path(run)/'summary.json'),artifact_sha256=collect_protocol_artifacts(run));atomic_write_json(path,ledger);return record
        for dataset in ['WN18RR','sports-copurchase','cloth-copurchase']:
            reference=references[dataset];metric='validation_accuracy' if dataset=='WN18RR' else 'validation_mrr'
            candidate=execute('validate_'+dataset,['scripts/run_convergence_candidate.py','--stage','wn' if dataset=='WN18RR' else 'lp','--dataset',dataset,'--reference-run',reference['run_directory']],'probe_last.pt' if dataset=='WN18RR' else 'link_head_last.pt')
            gain=100*(candidate['summary'][metric]-reference['summary'][metric])
            if gain<profile['minimum_validation_gain_pp']:
                ledger.setdefault('rejections',{})[dataset]={'validation_gain_pp':gain,'reason':'predeclared validation gate not met','new_test_evaluations':0};atomic_write_json(path,ledger);continue
            run=Path(candidate['run_directory']);s=candidate['summary'];cache=Path(s['embedding_cache']);checkpoints=Path(read_json_file(run/'checkpoints/location.json')['checkpoint_directory']);best=checkpoints/('probe_best_raw_linear.pt' if dataset=='WN18RR' else 'link_head_best.pt')
            lock={'dataset':dataset,'encoder_sha256':profile['encoder_sha256'],'classifier_checkpoint':str(best),'classifier_sha256':fingerprint(best),'validation_run_directory':str(run),'validation_summary_sha256':fingerprint(run/'summary.json'),'embedding_cache':str(cache),'inference_state_sha256':fingerprint(cache/'inference_state.json'),'locked_before_test':True,'test_used_for_selection':False,'selected_validation_'+('accuracy' if dataset=='WN18RR' else 'mrr'):s[metric],'graph_policy':s['graph_policy']}
            if dataset=='WN18RR':lock['feature_transform']='train_pair_affine'
            else:lock['head_variant']=s['head_variant']
            lock_path=logdir/(dataset+'_selection_lock.json')
            if lock_path.exists() and read_json_file(lock_path)!=lock:raise ValueError('Convergence selection lock changed')
            atomic_write_json(lock_path,lock)
            execute('confirm_'+dataset,['-m','came_repro.kg_selected_confirmation' if dataset=='WN18RR' else 'came_repro.protocol_repair',* ([] if dataset=='WN18RR' else ['--stage','confirm_lp']),'--selection-lock',str(lock_path)])
        ledger.update(state='completed_bounded_convergence_screen',all_unmet_repaired=False,remaining_protocol_review=['books-lp'],formal_repeats_by_user=True);atomic_write_json(path,ledger)
    except BaseException as error:
        if ledger['state']!='paused':ledger.update(state='failed',error=str(error));atomic_write_json(path,ledger)
        raise
    finally:supervisor_lock.close()


if __name__=='__main__':main()
