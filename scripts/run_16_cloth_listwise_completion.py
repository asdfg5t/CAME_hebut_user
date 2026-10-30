"""离线等待既定Cloth排序对照；验证及一次测试达标才接续固定十seed。"""
import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
if __name__=='__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
OUT=ROOT/'artifacts/cloth_listwise_completion'
LEDGER=ROOT/'artifacts/workflow_state/cloth_listwise_completion_current.json'
PAUSE=ROOT/'artifacts/workflow_state/cloth_listwise_completion_pause.json'
PRIOR=ROOT/'artifacts/workflow_state/remaining_objectives_20261018_current.json'
RELEASE=ROOT/'artifacts/delivery/cloth_listwise_completion_release.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_admission(prior):
    """Exactly the previously declared validation gate; no new recipe search."""
    from remaining_objective_backend_20261018 import verify_plan
    plan=verify_plan()
    record=prior['records'].get('validate_cloth',{})
    if record.get('state')!='completed':return False,'validation_incomplete'
    s=record['summary']
    if s['seed']!=0 or s['test_evaluations']!=0:raise ValueError('Fixed validation-only seed0 required')
    if 100*(s['validation_mrr']-plan['reference_validation']['cloth'])<plan['minimum_validation_gain_pp']-1e-10:
        return False,'validation_gain_not_met'
    test=prior['records'].get('confirm_cloth',{})
    if test.get('state')!='completed':return False,'confirmation_incomplete'
    return True,'eligible_for_independent_confirmation_audit'


def verify():
    from came_repro.author_loader import fingerprint,verify_originals
    from remaining_delivery import verify_release
    from remaining_objective_backend_20261018 import verify_plan
    verify_release();verify_plan();verify_originals()
    for path,digest in read(RELEASE)['files'].items():
        if fingerprint(ROOT/path)!=digest:raise ValueError('Frozen continuation input changed: '+path)


def summarize(ledger):
    # Reuse only the already tested reporting function, in this CPU controller.
    # Its process-local output directory is restored immediately; old files stay untouched.
    import cloth_target_delivery as report
    original=report.OUT;report.OUT=OUT
    try:report.summarize(ledger,dict(paper_mrr_pct=28.88,target_absolute_gap_pp=2.))
    finally:report.OUT=original


def pipeline():
    import psutil
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.pause_control import TrainingPaused
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.storage_budget import check_budget
    from cloth_wn_autonomous import artifact_hashes
    from remaining_delivery import audit_test
    verify();OUT.mkdir(parents=True,exist_ok=True)
    guard=acquire_gpu_lock(ROOT/'.runtime/cloth_listwise_completion.lock')
    ledger=read(LEDGER) if LEDGER.exists() else dict(release_sha256=fingerprint(RELEASE),state='waiting_existing_listwise',records={},results={},active_seconds=0.)
    logger=logging.getLogger('cloth_listwise_completion');logger.setLevel(logging.INFO)
    handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'pipeline.log',encoding='utf-8')]
    for h in handlers:
        fmt=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s');fmt.converter=time.gmtime;h.setFormatter(fmt);logger.addHandler(h)
    def persist():atomic_write_json(LEDGER,ledger)
    sleep_guard=False
    try:
        if ledger['release_sha256']!=fingerprint(RELEASE):raise ValueError('Do not overwrite an earlier release')
        if os.name=='nt':
            import ctypes
            sleep_guard=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        persist();logger.info('Wait existing fixed listwise candidate; no second GPU task, no test-driven recipe search')
        while True:
            if PAUSE.exists():raise TrainingPaused('Continuation explicitly paused')
            launch=read(ROOT/'artifacts/remaining_objectives_20261018/launch.json')
            try:p=psutil.Process(launch['pid']);alive=abs(p.create_time()-launch['creation_time'])<1
            except psutil.NoSuchProcess:alive=False
            if not alive:break
            time.sleep(15)
        prior=read(PRIOR);eligible,reason=validate_admission(prior)
        if not eligible:
            ledger.update(state='stopped_'+reason,prior_outcome=prior['state']);persist();return ledger
        validation=prior['records']['validate_cloth'];confirmed=prior['records']['confirm_cloth']
        for rec in [validation,confirmed]:
            for path,digest in rec['artifacts'].items():
                if fingerprint(path)!=digest:raise ValueError('Fixed candidate artifact changed')
        result=audit_test('cloth',confirmed['run_directory'])
        if result['official_test_queries']!=97456:raise ValueError('Complete official test required')
        result.update(seed=0,run_directory=confirmed['run_directory'],gap_pp=result['score_pct']-28.88,
            within_two_pp=abs(result['score_pct']-28.88)<=2.,seed0_training_reused=True)
        if '0' in ledger['results'] and ledger['results']['0']!=result:raise ValueError('Do not replace seed0 confirmation')
        ledger['results']['0']=result
        ledger['admission']=dict(passed=result['within_two_pp'],seed0_score_pct=result['score_pct'],recipe_changed_after_test=False,
            old_validation_half_pp_gate_retained=True,conditional_repetition_requested_by_user=True)
        if not result['within_two_pp']:
            ledger['state']='stopped_target_not_met_no_strategy_search';persist();summarize(ledger);return ledger
        run=Path(validation['run_directory']);s=validation['summary']
        recipe={key:s[key] for key in ['head_variant','lr','batch_size','max_epochs','patience','step_lr_size','step_lr_gamma','loss_reduction','head_training_dropout']}
        selected=dict(tasks=dict(cloth=dict(dataset='cloth-copurchase',validation_run=str(run),embedding_cache=s['embedding_cache'],
            graph_policy=s['graph_policy'],training_kind='listwise',recipe=recipe)),validation_candidate_sha256=fingerprint(run/'summary.json'),
            historical_tests_seen=True,test_did_not_select_recipe=True)
        selection=OUT/'protocol_selection.json'
        if selection.exists() and read(selection)!=selected:raise ValueError('Selection frozen before repeats')
        atomic_write_json(selection,selected);persist();summarize(ledger)
        check_budget(additional_project_bytes=128*2**20,additional_cache_bytes=2**30)
        worker=str(ROOT/'scripts/remaining_delivery_job.py')
        def execute(name,args,filename):
            verify()
            if PAUSE.exists():raise TrainingPaused('Continuation explicitly paused')
            rec=ledger['records'].setdefault(name,dict(state='ready',attempts=[]))
            if rec['state']=='completed':
                for path,digest in rec['artifacts'].items():
                    if fingerprint(path)!=digest:raise ValueError('Completed repetition artifact changed')
                return rec
            parent=best_resume(rec,filename)
            if name.startswith('test_') and not parent:
                for attempt in rec['attempts']:
                    old=attempt.get('run_directory')
                    if old and (Path(old)/'test_link_ranks.npz').exists():raise ValueError('Test output without consistent cursor requires audit')
            if parent:args+=['--resume-head',str(parent)]
            ledger['state']='running_'+name;persist();logger.info('JOB %s parent=%s',name,parent)
            code,directory=run_child([worker,*args],PAUSE,ledger,LEDGER,rec,logger,float('inf'))
            if code==75 or PAUSE.exists():raise TrainingPaused('Full epoch/test cursor saved')
            if code or not directory:raise RuntimeError('Child failed: '+name+' '+str(code))
            rec.update(state='completed',run_directory=directory,summary=read(Path(directory)/'summary.json'),artifacts=artifact_hashes(directory));persist();return rec
        for seed in range(1,10):
            key=str(seed)
            if key in ledger['results']:
                for path,digest in ledger['records']['test_s'+key]['artifacts'].items():
                    if fingerprint(path)!=digest:raise ValueError('Saved test result changed')
                continue
            record=execute('train_s'+key,['--stage','train','--task','cloth','--seed',key,'--selection',str(selection)],'link_head_last.pt')
            directory=Path(record['run_directory']);summary=record['summary'];cache=Path(summary['embedding_cache'])
            head=Path(read(directory/'checkpoints/location.json')['checkpoint_directory'])/'link_head_best.pt'
            lock=dict(dataset='cloth-copurchase',encoder_sha256=s['encoder_sha256'],classifier_checkpoint=str(head),classifier_sha256=fingerprint(head),
                validation_run_directory=str(directory),validation_summary_sha256=fingerprint(directory/'summary.json'),embedding_cache=str(cache),
                inference_state_sha256=fingerprint(cache/'inference_state.json'),locked_before_test=True,test_used_for_selection=False,
                graph_policy=s['graph_policy'],head_variant=summary['head_variant'],selected_validation_mrr=summary['validation_mrr'])
            file=OUT/('seed'+key+'_test_lock.json')
            if file.exists() and read(file)!=lock:raise ValueError('Frozen test head changed')
            atomic_write_json(file,lock)
            tested=execute('test_s'+key,['--stage','test','--task','cloth','--selection',str(file)],'test_cursor.json')
            score=audit_test('cloth',tested['run_directory']);score.update(seed=seed,run_directory=tested['run_directory'],gap_pp=score['score_pct']-28.88,
                within_two_pp=abs(score['score_pct']-28.88)<=2.,seed0_training_reused=False)
            ledger['results'][key]=score;persist();summarize(ledger)
            logger.info('RESULT seed=%d MRR=%.8f gap=%+.8fpp; all seeds retained',seed,score['score_pct'],score['gap_pp'])
        ledger['state']='completed_ten_seeds';persist();summarize(ledger);return ledger
    except BaseException as error:
        ledger.update(state='paused' if isinstance(error,(TrainingPaused,KeyboardInterrupt)) or PAUSE.exists() else 'failed',error=str(error));persist();raise
    finally:
        if sleep_guard:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for h in handlers:h.close();logger.removeHandler(h)
        guard.close()


def main():
    from came_repro.atomic_io import atomic_write_json
    from came_repro.ide_workflow import acquire_gpu_lock
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--detach',action='store_true');parser.add_argument('--pause',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    for stream in [sys.stdout,sys.stderr]:
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    if args.pause:
        atomic_write_json(PAUSE,dict(reason='user_requested_listwise_completion_pause'));return
    verify();OUT.mkdir(parents=True,exist_ok=True)
    if args.dry_run:print('Frozen continuation verified; no GPU jobs started.');return
    if PAUSE.exists():
        if not args.resume:raise RuntimeError('Explicit pause retained; use --resume for this continuation only')
        guard=acquire_gpu_lock(ROOT/'.runtime/cloth_listwise_completion.lock')
        try:
            atomic_write_json(OUT/('pause_history_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),read(PAUSE));PAUSE.unlink()
        finally:guard.close()
    if not args.detach:pipeline();return
    import subprocess,psutil
    guard=acquire_gpu_lock(ROOT/'.runtime/cloth_listwise_completion_launch.lock')
    try:
        file=OUT/'launch.json'
        if file.exists():
            old=read(file)
            try:
                p=psutil.Process(old['pid'])
                if abs(p.create_time()-old['creation_time'])<1:print('Already running; no duplicate launched.');return
            except psutil.NoSuchProcess:pass
        log=OUT/('console_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.log')
        with log.open('ab') as output:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                close_fds=True,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB,
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'})
        atomic_write_json(file,dict(pid=child.pid,creation_time=psutil.Process(child.pid).create_time(),console_log=str(log),one_shot_not_recurring=True));print('Started offline continuation PID='+str(child.pid))
    finally:guard.close()


if __name__=='__main__':main()
