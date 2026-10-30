"""Products: wait for Goodreads, measure real CUDA speed, then bounded shared ten-head run."""
import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
if __name__=='__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
OUT=ROOT/'artifacts/products_sampled_stream'
PROFILE=ROOT/'configs/products_sampled_stream_20261021.json'
RELEASE=ROOT/'artifacts/delivery/products_sampled_stream_release.json'
LEDGER=ROOT/'artifacts/workflow_state/products_sampled_stream_current.json'
PAUSE=ROOT/'artifacts/workflow_state/products_sampled_stream_pause.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify():
    from came_repro.author_loader import fingerprint,verify_originals
    verify_originals()
    for path,digest in read(RELEASE)['input_sha256'].items():
        if fingerprint(path)!=digest:
            raise ValueError('Frozen Products input changed: '+path)
    p=read(PROFILE)
    if (p['nodes']!=2449029 or p['test_nodes']!=2213091 or p['width']!=1024 or p['precision']!='fp32'
            or p['head']['seeds']!=list(range(10)) or p['inference']['fanouts']!=[15,15]
            or p['input_cache']['maximum_bytes']!=4*2**30):
        raise ValueError('Declared complete official Products protocol changed')
    return p


def artifact_hashes(run):
    from came_repro.author_loader import fingerprint
    run=Path(run)
    paths=[run/n for n in ['config.json','manifest.json','status.json','summary.json','logs/train.log','logs/metrics.jsonl','logs/metrics.csv']]
    paths+=list(run.glob('*.npz'))+list((run/'tensorboard').glob('*'))+list((run/'figures').glob('*.png'))
    location=run/'checkpoints/location.json'
    if location.exists():
        paths+=[location]+list(Path(read(location)['checkpoint_directory']).glob('*'))
    return {str(x):fingerprint(x) for x in paths if x.is_file()}


def verify_completed(record):
    from came_repro.author_loader import fingerprint
    for path,digest in record['artifacts'].items():
        if fingerprint(path)!=digest:
            raise ValueError('Completed Products artifact changed: '+path)


def audit_complete(run):
    import numpy as np
    from came_repro.products_sampled_stream import metadata
    run=Path(run)
    s=read(run/'summary.json')
    splits,labels=metadata(read(PROFILE))
    with np.load(run/'all_seeds_test_predictions.npz') as values:
        np.testing.assert_array_equal(values['node_ids'],splits['test'])
        np.testing.assert_array_equal(values['labels'],labels[splits['test']])
        np.testing.assert_array_equal(values['seeds'],np.arange(10))
        predictions=values['prediction']
        if predictions.shape!=(2213091,10) or predictions.min()<0 or predictions.max()>46:
            raise ValueError('All ten complete official prediction columns required')
        scores=np.mean(predictions==values['labels'][:,None],axis=0)
    np.testing.assert_array_equal(scores,s['test_accuracy'])
    if (s['test_evaluations_per_seed']!=1 or s['mean_pct']!=float(scores.mean()*100)
            or s['sample_sd_pp']!=float(scores.std(ddof=1)*100)):
        raise ValueError('Complete test statistics do not match saved predictions')
    cursor=read(s['complete_cursor'])
    if cursor['cursor']!=2213091:
        raise ValueError('Official test cursor incomplete')
    return dict(independently_recomputed=True,complete_official_test_nodes=2213091,
        seeds=list(range(10)),test_accuracy=scores.tolist(),mean_pct=s['mean_pct'],
        sample_sd_pp=s['sample_sd_pp'],paper_pct=s['paper_pct'],gap_pp=s['gap_pp'],
        inference_is_declared_approximation=True,run_directory=str(run),paper_reproduction_complete=False)


def main():
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.pause_control import TrainingPaused
    from came_repro.quality_pipeline import run_child,best_resume
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--pause',action='store_true')
    parser.add_argument('--extend-hours',type=float,default=0)
    parser.add_argument('--stage',choices=['calibration','features','heads','test'])
    parser.add_argument('--resume-head',type=Path)
    parser.add_argument('--feature-run',type=Path)
    parser.add_argument('--head-run',type=Path)
    args=parser.parse_args()
    if args.pause:
        atomic_write_json(PAUSE,dict(reason='user_paused_Products_sampled_stream'))
        print('已请求在完整行/完整线性头更新边界保存，后续同文件恢复。',flush=True)
        return
    p=verify()
    if not 0<=args.extend_hours<=24:
        raise ValueError('Explicit extension must be between0 and24 hours')
    if args.dry_run:
        from came_repro.products_sampled_stream import estimate_seconds
        threshold=p['nodes']*p['budget']['inference_safety_factor']/(7200-p['budget']['head_and_setup_reserve_seconds'])
        print(json.dumps(dict(stage='dry_run_only',GPU_started=False,full_official_test_nodes=p['test_nodes'],
            fanouts=p['inference']['fanouts'],head_seeds=p['head']['seeds'],new_feature_bytes=235938*1024*4,
            minimum_measured_nodes_per_second=threshold,actual_throughput=None,test_accuracy=None,
            inference_is_declared_approximation=True),ensure_ascii=False,indent=2),flush=True)
        return
    if args.stage:
        import torch
        from came_repro.products_memory import require_cuda as check_cuda
        from came_repro import products_sampled_stream as core
        check_cuda()
        torch.set_num_threads(4)
        if args.stage!='calibration':
            receipt=read(OUT/'acceptance_lock.json')
            if (receipt['release_sha256']!=fingerprint(RELEASE)
                    or fingerprint(receipt['summary_path'])!=receipt['summary_sha256'] or not read(receipt['summary_path'])['passed']):
                raise ValueError('Matching actual CUDA/numeric/speed acceptance required')
        if args.stage=='calibration':core.calibration(p)
        elif args.stage=='features':core.features(p,args.resume_head)
        elif args.stage=='heads':core.fit_heads(p,args.feature_run,args.resume_head)
        else:core.test_heads(p,args.head_run,OUT/'test_selection_lock.json',args.resume_head)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    own=acquire_gpu_lock(ROOT/'.runtime/products_sampled_stream.lock')
    workflow=None
    logger=logging.getLogger('products_sampled_stream')
    logger.setLevel(logging.INFO)
    handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'pipeline.log',encoding='utf-8')]
    for handler in handlers:
        formatter=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s')
        formatter.converter=time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    ledger=read(LEDGER) if LEDGER.exists() else dict(state='ready_for_runtime_calibration',release_sha256=fingerprint(RELEASE),
        active_seconds=0.,allowed_active_seconds=7200.,records={})
    sleep_guard=False
    def persist():atomic_write_json(LEDGER,ledger)
    try:
        if ledger['release_sha256']!=fingerprint(RELEASE):
            raise ValueError('Existing Products experiment identity changed')
        if args.extend_hours:
            ledger['allowed_active_seconds']+=args.extend_hours*3600
            ledger.setdefault('explicit_budget_extensions',[]).append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hours=args.extend_hours))
            persist()
        if ledger['state']=='completed_ten_seeds_sampled_full_test':
            for rec in ledger['records'].values():verify_completed(rec)
            print(json.dumps(ledger['result'],ensure_ascii=False),flush=True)
            return
        if ledger['active_seconds']>=ledger['allowed_active_seconds']:
            raise TrainingPaused('累计两小时预算已用尽；检查断点后用--extend-hours显式增加预算，重启不会重置。')
        if PAUSE.exists():
            history=OUT/'pause_history'
            history.mkdir(exist_ok=True)
            atomic_write_json(history/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),read(PAUSE))
            PAUSE.unlink()  # Only this entry's pause; other workflows are untouched.
        while workflow is None:
            if PAUSE.exists():raise TrainingPaused('User paused while waiting; no GPU started')
            try:workflow=acquire_gpu_lock(ROOT/'.runtime/goodreads_official.lock')
            except RuntimeError:
                if ledger['state']!='waiting_for_Goodreads_workflow':
                    ledger['state']='waiting_for_Goodreads_workflow'
                    persist()
                    logger.info('WAIT: 用户Goodreads完整workflow仍在运行；不在两个GPU子阶段之间插队。')
                time.sleep(10)
        probe=acquire_gpu_lock()
        probe.close()
        import ctypes
        sleep_guard=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        from came_repro.products_controller_storage import reserve_external
        reserve_external(p['budget']['extra_project_bytes'],p['budget']['extra_cache_bytes'])
        def execute(stage,feature_run=None,head_run=None):
            verify()
            rec=ledger['records'].setdefault(stage,dict(state='ready',attempts=[]))
            if rec['state']=='completed':
                verify_completed(rec)
                return rec
            remaining=ledger['allowed_active_seconds']-ledger['active_seconds']
            if remaining<=0 or PAUSE.exists():raise TrainingPaused('Products active budget/pause boundary')
            filename={'features':'feature_cursor.json','heads':'heads_last.pt','test':'predictions_cursor.json'}.get(stage)
            parent=best_resume(rec,filename) if filename else None
            arguments=[str(Path(__file__)),'--stage',stage]
            if parent:arguments+=['--resume-head',str(parent)]
            if feature_run:arguments+=['--feature-run',str(feature_run)]
            if head_run:arguments+=['--head-run',str(head_run)]
            ledger['state']='running_'+stage
            logger.info('PRODUCTS_STAGE %s parent=%s remaining_active_minutes=%.1f',stage,parent,remaining/60)
            code,run=run_child(arguments,PAUSE,ledger,LEDGER,rec,logger,remaining)
            if code==75 or PAUSE.exists():
                rec.update(state='paused',run_directory=run)
                reason=read(Path(run)/'status.json').get('error','') if run and (Path(run)/'status.json').exists() else ''
                raise TrainingPaused('Products durable boundary saved; '+reason)
            if code or not run:raise RuntimeError('Products stage failed: %s exit=%s'%(stage,code))
            rec.update(state='completed',run_directory=run,summary=read(Path(run)/'summary.json'),artifacts=artifact_hashes(run))
            persist()
            return rec
        accepted=execute('calibration')
        if not accepted['summary']['passed']:
            ledger['state']='calibration_rejected_no_full_workload_started'
            persist()
            logger.error('真实校准未达到两小时/数值保真门槛：%s。未缩官方测试集、未启动完整任务。',json.dumps(accepted['summary'],ensure_ascii=False))
            return
        receipt=dict(summary_path=str(Path(accepted['run_directory'])/'summary.json'),
            summary_sha256=fingerprint(Path(accepted['run_directory'])/'summary.json'),release_sha256=fingerprint(RELEASE))
        if (OUT/'acceptance_lock.json').exists() and read(OUT/'acceptance_lock.json')!=receipt:
            raise ValueError('Acceptance lock cannot be replaced')
        atomic_write_json(OUT/'acceptance_lock.json',receipt)
        if args.prepare_only:
            ledger['state']='CUDA_and_speed_accepted_waiting_user'
            persist()
            return
        feature=execute('features')
        trained=execute('heads',feature_run=feature['run_directory'])
        h=trained['summary']
        lock=dict(locked_before_test=True,test_used_for_selection=False,head_run_directory=trained['run_directory'],
            head_summary_sha256=fingerprint(Path(trained['run_directory'])/'summary.json'),classifier_sha256=h['classifier_sha256'],
            all_seeds_retained=True,release_sha256=fingerprint(RELEASE))
        path=OUT/'test_selection_lock.json'
        if path.exists() and read(path)!=lock:raise ValueError('Refuse to replace selected heads after test')
        atomic_write_json(path,lock)
        tested=execute('test',head_run=trained['run_directory'])
        result=audit_complete(tested['run_directory'])
        ledger.update(state='completed_ten_seeds_sampled_full_test',result=result)
        persist()
        atomic_write_json(OUT/'results.json',result)
        import csv
        with (OUT/'results.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['seed','test_accuracy_pct','paper_pct','gap_pp'])
            for seed,score in enumerate(result['test_accuracy']):writer.writerow([seed,score*100,result['paper_pct'],score*100-result['paper_pct']])
        logger.info('INDEPENDENT_FULL_TEST_AUDIT %s',json.dumps(result,ensure_ascii=False))
    except TrainingPaused as error:
        ledger.update(state='paused',error=str(error))
        persist()
        logger.info('SAFE_PAUSE: %s；保留已有产物。后续运行同一个文件继续。',error)
        return
    except BaseException as error:
        ledger.update(state='paused' if isinstance(error,(TrainingPaused,KeyboardInterrupt)) else 'failed',error=str(error))
        persist()
        raise
    finally:
        if sleep_guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for handler in handlers:handler.close();logger.removeHandler(handler)
        if workflow:workflow.close()
        own.close()


if __name__=='__main__':main()
