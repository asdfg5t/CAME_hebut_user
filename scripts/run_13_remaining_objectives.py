"""Finite serial WN/Cloth validation after the current Products diagnostic exits."""
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'scripts'))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
OUT=ROOT/'artifacts/remaining_objectives_20261018'
PROFILE=ROOT/'configs/remaining_objectives_20261018.json'
LEDGER=ROOT/'artifacts/workflow_state/remaining_objectives_20261018_current.json'
PAUSE=ROOT/'artifacts/workflow_state/all_unresolved_20261018_pause.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pipeline():
    import psutil
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.ide_workflow import acquire_gpu_lock
    from remaining_objective_backend_20261018 import verify_plan
    plan=verify_plan();OUT.mkdir(parents=True,exist_ok=True)
    identity=dict(profile_sha256=fingerprint(PROFILE),dependencies=plan['dependencies'])
    guard=acquire_gpu_lock(ROOT/'.runtime/remaining_objectives_20261018.lock')
    ledger=read(LEDGER) if LEDGER.exists() else dict(identity=identity,state='waiting_for_products_exit',records={},active_seconds=0.,results={},rejections={})
    if ledger['identity']!=identity:guard.close();raise ValueError('Objective queue identity changed')
    def persist():atomic_write_json(LEDGER,ledger)
    persist();sleep_guard=False
    if os.name=='nt':
        import ctypes
        sleep_guard=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    logger=logging.getLogger('remaining_objectives');logger.setLevel(logging.INFO)
    handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'pipeline.log',encoding='utf-8')]
    for h in handlers:
        fmt=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s');fmt.converter=time.gmtime;h.setFormatter(fmt);logger.addHandler(h)
    try:
        while True:
            if PAUSE.exists():raise RuntimeError('Explicit user/queue pause retained')
            launch=read(ROOT/'artifacts/products_csc64_20261018/launch.json')
            try:p=psutil.Process(launch['pid']);alive=abs(p.create_time()-launch['creation_time'])<1
            except psutil.NoSuchProcess:alive=False
            if not alive:
                try:probe=acquire_gpu_lock();probe.close();break
                except RuntimeError:pass
            time.sleep(15)
        # Products may have failed. Do not restart it; only independent tasks follow.
        ledger['products_predecessor_outcome']=read(ROOT/'artifacts/workflow_state/products_csc64_20261018_current.json')['state'];persist()
        from came_repro.quality_pipeline import run_child,best_resume
        from came_repro.pause_control import TrainingPaused
        from came_repro.storage_budget import check_budget
        from cloth_wn_autonomous import artifact_hashes,independent_wn
        from audit_unresolved_validation import audit_validation
        check_budget(additional_project_bytes=128*2**20,additional_cache_bytes=256*2**20)
        def execute(name,arguments,checkpoint=None):
            verify_plan()
            if PAUSE.exists():raise TrainingPaused('Explicit pause retained')
            record=ledger['records'].setdefault(name,dict(state='ready',attempts=[]))
            if record['state']=='completed':
                for path,sha in record['artifacts'].items():
                    if fingerprint(path)!=sha:raise ValueError('Completed objective artifact changed')
                return record
            if name.startswith('confirm_') and record['attempts']:raise ValueError('Interrupted confirmation requires audit; no automatic retest')
            parent=best_resume(record,checkpoint) if checkpoint else None
            if parent:arguments+=['--resume-head',str(parent)]
            prior=read(ROOT/'artifacts/workflow_state/all_unresolved_20261018_current.json')['active_seconds']
            prior+=read(ROOT/'artifacts/workflow_state/products_csc64_20261018_current.json')['active_seconds']
            remaining=min(plan['maximum_new_active_hours']*3600-ledger['active_seconds'],24*3600-prior-ledger['active_seconds'])
            if remaining<=0:raise TrainingPaused('Combined finite activity budget exhausted')
            ledger['state']='running_'+name;persist();logger.info('STAGE %s parent=%s',name,parent)
            code,run=run_child(arguments,PAUSE,ledger,LEDGER,record,logger,remaining)
            if code==75 or PAUSE.exists():raise TrainingPaused('Durable boundary saved')
            if code:record.update(state='failed',exit_code=code);persist();raise RuntimeError('Objective stage failed: '+name+' '+str(code))
            summary=read(OUT/'acceptance.json') if name=='acceptance' else read(Path(run)/'summary.json')
            artifacts={str(OUT/'acceptance.json'):fingerprint(OUT/'acceptance.json')} if name=='acceptance' else artifact_hashes(run)
            record.update(state='completed',run_directory=run,summary=summary,artifacts=artifacts);persist();return record
        backend=str(ROOT/'scripts/remaining_objective_backend_20261018.py')
        execute('acceptance',[backend,'--stage','acceptance'])
        for task in ('wn','cloth'):
            item=execute('validate_'+task,[backend,'--stage',task],'probe_last.pt' if task=='wn' else 'link_head_last.pt')
            summary=item['summary'];run=Path(item['run_directory']);audit=audit_validation(run,task)
            if summary['seed']!=0 or summary['test_evaluations']!=0:
                raise ValueError('Fixed seed0 validation-only selection required')
            atomic_write_json(OUT/(task+'_validation_audit.json'),audit)
            score=audit['validation_score'];gain=100*(score-plan['reference_validation'][task])
            if gain<plan['minimum_validation_gain_pp']-1e-10:
                ledger['rejections'][task]=dict(validation_score=score,gain_pp=gain,new_test_evaluations=0);persist();continue
            directory=Path(read(run/'checkpoints/location.json')['checkpoint_directory'])
            head=directory/('link_head_best.pt' if task=='cloth' else 'probe_best_raw_linear.pt')
            config=read(run/'config.json');cache=Path(summary.get('embedding_cache') or config['evaluation']['embedding_cache'])
            lock=dict(dataset=summary['dataset'],encoder_sha256=plan['encoder_sha256'],classifier_checkpoint=str(head),classifier_sha256=fingerprint(head),
                validation_run_directory=str(run),validation_summary_sha256=fingerprint(run/'summary.json'),embedding_cache=str(cache),
                inference_state_sha256=fingerprint(cache/'inference_state.json'),locked_before_test=True,test_used_for_selection=False,graph_policy=summary['graph_policy'])
            if task=='cloth':lock.update(head_variant=summary['head_variant'],selected_validation_mrr=score)
            else:lock.update(feature_transform='train_pair_affine',selected_validation_accuracy=score)
            file=OUT/(task+'_selection_lock.json')
            if file.exists() and read(file)!=lock:raise ValueError('Locked selection changed')
            atomic_write_json(file,lock)
            args=['-m','came_repro.protocol_repair','--stage','confirm_lp','--selection-lock',str(file)] if task=='cloth' else ['-m','came_repro.kg_selected_confirmation','--selection-lock',str(file)]
            confirmed=execute('confirm_'+task,args)
            if task=='cloth':
                from audit_two_lp_confirmation import audit_confirmation
                result=audit_confirmation(Path(confirmed['run_directory']),file)
            else:result=independent_wn(Path(confirmed['run_directory']),file)
            ledger['results'][task]=result;persist()
        ledger['state']='completed_bounded_objectives_pending_visual_review';persist()
    except BaseException as error:
        ledger.update(state='paused' if PAUSE.exists() or type(error).__name__=='TrainingPaused' else 'failed',error=str(error));persist();raise
    finally:
        if sleep_guard:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for h in handlers:h.close();logger.removeHandler(h)
        guard.close()


def main():
    from remaining_objective_backend_20261018 import verify_plan
    verify_plan()
    # Reuse only the pinned hidden-process launcher; all workflow logic is above.
    import run_11_products_block64_repair as relay
    relay.__file__=str(Path(__file__).resolve());relay.OUT=OUT;relay.PROFILE=PROFILE;relay.LEDGER=LEDGER;relay.pipeline=pipeline
    relay.main()


if __name__=='__main__':main()
