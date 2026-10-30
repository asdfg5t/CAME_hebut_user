"""PyCharm one-click repeats for independently verified tasks only; currently FB15K237."""
import datetime
import logging
import math
import sys
from pathlib import Path

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def verified_seed0(profile,summary):
    if (summary.get('dataset')!='FB15K237' or summary.get('seed')!=0 or summary.get('test_evaluations')!=1
        or summary.get('graph_policy')!='train_reverse' or summary.get('feature_transform','raw')!='raw'
        or summary.get('encoder_sha256')!=profile['encoder_sha256']):
        raise ValueError('Bound, complete raw-linear FB seed0 result required')
    value=100*summary['test_accuracy'];gap=value-profile['paper_accuracy_pct']
    if not math.isfinite(value) or abs(gap)>profile['preferred_absolute_gap_pp']:
        raise ValueError('Task is not independently confirmed within the stated gap')
    return value,gap


def main():
    import argparse
    import json
    import numpy as np
    from came_repro.atomic_io import read_json_file,atomic_write_json
    from came_repro.author_loader import fingerprint,verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.all_unmet_repair_pipeline import collect_protocol_artifacts
    from came_repro.prepared_data import prepared_directory
    from came_repro.storage_budget import check_budget
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    profile_path=ROOT/'configs/validated_remaining_tasks_20261014.json';profile=read_json_file(profile_path)
    seed0_run=ROOT/profile['seed0_confirmed_run'];summary0=read_json_file(seed0_run/'summary.json')
    value,gap=verified_seed0(profile,summary0)
    saved=np.load(seed0_run/'test_predictions.npz');official=np.load(prepared_directory('FB15K237')/'test_triples.npy',mmap_mode='r')
    if not np.array_equal(saved['pairs'],official[:,[0,2]]) or not np.array_equal(saved['labels'],official[:,1]) or float(np.mean(saved['prediction']==saved['labels']))!=summary0['test_accuracy']:
        raise ValueError('Seed0 saved predictions are not the complete official test')
    lock0=read_json_file(ROOT/profile['seed0_selection_lock'])
    if fingerprint(lock0['classifier_checkpoint'])!=lock0['classifier_sha256'] or fingerprint(ROOT/profile['seed0_selection_lock'])!=summary0['selection_lock_sha256']:
        raise ValueError('Protected seed0 selection identity changed')
    sources={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}
    identity={'profile_sha256':fingerprint(profile_path),'script_sha256':fingerprint(__file__),'sources':sources,'seed0_summary_sha256':fingerprint(seed0_run/'summary.json'),'seed0_prediction_sha256':fingerprint(seed0_run/'test_predictions.npz'),'seed0_lock_sha256':fingerprint(ROOT/profile['seed0_selection_lock'])}
    path=ROOT/'artifacts/workflow_state/validated_remaining_repeats_current.json';pause=ROOT/'artifacts/workflow_state/validated_remaining_repeats_pause.json'
    ledger=read_json_file(path) if path.exists() else {'identity':identity,'state':'ready_for_user_run','records':{},'active_seconds':0.,'seed0_reused':True,'formal_repeat_count':10,'new_seed_count':9,'test_score_used_to_filter_seeds':False}
    if ledger['identity']!=identity:raise ValueError('Validated repeat protocol changed; review before continuing')
    atomic_write_json(path,ledger)
    print('VERIFIED_READY: FB15K237 seed0=%.6f%%, paper=%.2f%%, gap=%+.6fpp; reuse seed0, train seeds1-9 only.'%(value,profile['paper_accuracy_pct'],gap),flush=True)
    if args.dry_run:
        print('DRY_RUN: 9 validation-training jobs +9 saved-head confirmations; GPU jobs started=0. Other tasks excluded.',flush=True);return
    if ledger['state']=='completed_ten_downstream_seeds':print('ALREADY_COMPLETED: all ten results retained; no retraining.',flush=True);return
    if pause.exists():
        print('PAUSE_FILE_PRESENT: preserve pause; remove the explicitly requested pause file when ready to resume.',flush=True);return
    # Refuse to overlap the ongoing diagnostic controllers or their next GPU children.
    for name in ['all_unmet_repair_current.json','convergence_repair_current.json','products_single_seed_current.json']:
        other=ROOT/'artifacts/workflow_state'/name
        if other.exists() and read_json_file(other).get('state') in ('running','encoding','validation_head_training','full_official_test','waiting_for_convergence_GPU_release'):
            print('GPU_DIAGNOSTICS_ACTIVE: this ready file has not started training. Click it after the current diagnostics finish.',flush=True);return
    try:guard=acquire_gpu_lock();guard.close()
    except RuntimeError:
        print('GPU_BUSY: existing work preserved; no repeat task started. Click again after it finishes.',flush=True);return
    controller=acquire_gpu_lock(ROOT/'.runtime/validated_remaining_repeats.lock')
    logger=logging.getLogger('validated_remaining_repeats');logger.setLevel(logging.INFO)
    directory=ROOT/'artifacts/validated_remaining_repeats';directory.mkdir(parents=True,exist_ok=True)
    for handler in [logging.StreamHandler(),logging.FileHandler(directory/('pipeline_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log'),encoding='utf-8')]:
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logger.addHandler(handler)
    try:
        verify_originals();check_budget(additional_cache_bytes=profile['resources']['checkpoint_metadata_reservation_bytes'])
        encoder=Path(read_json_file(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
        if fingerprint(encoder)!=profile['encoder_sha256']:raise ValueError('Confirmed frozen encoder changed')
        def execute(jid,arguments,resume_name=None):
            if fingerprint(profile_path)!=identity['profile_sha256'] or fingerprint(__file__)!=identity['script_sha256']:raise ValueError('Frozen validated repeat entry changed')
            for name,digest in sources.items():
                if fingerprint(ROOT/name)!=digest:raise ValueError('Frozen validated repeat source changed')
            record=ledger['records'].setdefault(jid,{'state':'ready','attempts':[]})
            if record['state']=='completed':
                for name,digest in record['artifact_sha256'].items():
                    if fingerprint(name)!=digest:raise ValueError('Completed repeat artifact changed')
                return record
            if resume_name:
                checkpoint=best_resume(record,resume_name)
                if checkpoint:arguments+=['--resume-head',str(checkpoint)]
            ledger['state']='running';code,run=run_child(arguments,pause,ledger,path,record,logger,float('inf'))
            if code or not run:
                ledger['state']=record['state']='paused' if code==75 else 'failed';atomic_write_json(path,ledger);raise RuntimeError('Validated repeat child failed '+jid)
            record.update(state='completed',run_directory=run,summary=read_json_file(Path(run)/'summary.json'),artifact_sha256=collect_protocol_artifacts(run));atomic_write_json(path,ledger);return record
        results=[summary0]
        for seed in profile['new_training_seeds']:
            record=execute('FB15K237_validation_seed'+str(seed),['-m','came_repro.kg_graph_probe','--dataset','FB15K237','--checkpoint',str(encoder),'--graph-policy','train_reverse','--seed',str(seed)],'probe_last.pt')
            run=Path(record['run_directory']);s=record['summary'];best=Path(read_json_file(run/'checkpoints/location.json')['checkpoint_directory'])/'probe_best.pt'
            selected={'dataset':'FB15K237','encoder_sha256':profile['encoder_sha256'],'graph_policy':'train_reverse','feature_transform':'raw','classifier_checkpoint':str(best),'classifier_sha256':fingerprint(best),'validation_run_directory':str(run),'validation_summary_sha256':fingerprint(run/'summary.json'),'selected_validation_accuracy':s['validation_accuracy'],'locked_before_test':True,'test_used_for_selection':False,'embedding_cache':lock0['embedding_cache']}
            lock_path=directory/('FB15K237_seed'+str(seed)+'_selection_lock.json')
            if lock_path.exists() and read_json_file(lock_path)!=selected:raise ValueError('Repeat saved-weight selection lock changed')
            atomic_write_json(lock_path,selected)
            confirmed=execute('FB15K237_confirmation_seed'+str(seed),['-m','came_repro.kg_selected_confirmation','--selection-lock',str(lock_path)])
            predictions=np.load(Path(confirmed['run_directory'])/'test_predictions.npz')
            if not np.array_equal(predictions['pairs'],official[:,[0,2]]) or not np.array_equal(predictions['labels'],official[:,1]) or float(np.mean(predictions['prediction']==predictions['labels']))!=confirmed['summary']['test_accuracy']:
                raise ValueError('Complete official repeat predictions/accuracy mismatch')
            results.append(confirmed['summary'])
            scores=np.array([r['test_accuracy']*100 for r in results])
            ledger['statistics']={'completed':len(scores),'expected':10,'all_seeds_included':True,'mean_accuracy_pct':float(scores.mean()),'sample_sd_pp':float(scores.std(ddof=1)),'paper_accuracy_pct':94.59,'gap_pp':float(scores.mean()-94.59),'scores_pct':scores.tolist(),'complete_repeat_grid':len(scores)==10,'pretraining_checkpoints':1}
            atomic_write_json(path,ledger);logger.info('FB progress=%d/10 mean=%.6f%% SD=%.6fpp',len(scores),scores.mean(),scores.std(ddof=1))
        ledger['state']='completed_ten_downstream_seeds';atomic_write_json(path,ledger)
        atomic_write_json(directory/'ten_seed_results.json',{'scope':profile['scope'],'statistics':ledger['statistics'],'all_seed_summaries':results,'first_four_not_retrained':True,'paper_reproduction_complete':False})
    finally:controller.close()


if __name__=='__main__':main()
