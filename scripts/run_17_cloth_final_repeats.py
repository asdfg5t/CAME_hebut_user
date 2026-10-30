"""Cloth作业08最终入口：验收后完成十seed，不再按论文差距阻止训练。"""
import argparse
import copy
import csv
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
OUT=ROOT/'artifacts/cloth_final_repeats'
PROFILE=ROOT/'configs/cloth_final_repeats_20261019.json'
RELEASE=ROOT/'artifacts/delivery/cloth_final_repeats_release.json'
LEDGER=ROOT/'artifacts/workflow_state/cloth_final_repeats_current.json'
PAUSE=ROOT/'artifacts/workflow_state/cloth_final_repeats_pause.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_profile(p):
    expected=dict(schema='came_cloth_final_fixed_repeats_v1',dataset='cloth-copurchase',nodes=125839,width=1024,
        official_validation_queries=121323,official_test_queries=97456,fixed_negative_candidates=150,
        maximum_new_seed0_tests=0,training_kind='dropout_bce',metric_admission_gate=False)
    if any(p.get(k)!=v for k,v in expected.items()) or p.get('seeds')!=list(range(10)):
        raise ValueError('必须保留Cloth完整数据、全部固定seed、复用seed0且不按测试分数准入。')
    if p['recipe']!=dict(head_variant='two_hidden_layernorm_head_upstream_schedule',lr=.001,batch_size=4096,
            max_epochs=300,patience=20,step_lr_size=None,step_lr_gamma=None,loss_reduction='separate_mean_sum',head_training_dropout=.3):
        raise ValueError('正式重复方案必须与锁定验证方案完全相同。')


def verify():
    from came_repro.author_loader import fingerprint,verify_originals
    from remaining_delivery import verify_release
    verify_release();verify_originals();p=read(PROFILE);validate_profile(p)
    for path,digest in read(RELEASE)['files'].items():
        file=Path(path);file=file if file.is_absolute() else ROOT/file
        if fingerprint(file)!=digest:raise ValueError('Cloth正式重复冻结输入改变：'+path)
    selection=read(ROOT/p['selection_file'])['tasks']['cloth']
    if selection['training_kind']!=p['training_kind'] or selection['recipe']!=p['recipe']:
        raise ValueError('Selected validation recipe changed')
    previous=read(ROOT/p['existing_preflight_ledger'])
    receipt=previous['records']['preflight']
    if receipt['state']!='completed' or not receipt['summary']['passed'] or not receipt['summary']['full_optimizer_RNG_sampler_history_resume_verified']:
        raise ValueError('完整实际CUDA/恢复验收必须存在')
    for file,digest in receipt['artifacts'].items():
        if fingerprint(file)!=digest:raise ValueError('Previous verified acceptance changed')
    return p


def check_gpu():
    """Runtime check of fixed head/cache on64 validation rows; no new test."""
    import numpy as np
    import torch
    from came_repro.protocol_repair import check_cuda
    from came_repro.experiment import Experiment,seed_everything
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.lp_validation_repair import load_verified_cache,make_head
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro.task_metrics import ranking_metrics
    from came_repro import link_evaluation as link
    p=verify();check_cuda();torch.set_num_threads(4);seed_everything(0)
    item=read(ROOT/p['selection_file'])['tasks']['cloth'];run=Path(item['validation_run']);s=read(run/'summary.json')
    cache=Path(item['embedding_cache']);gpu,audit=load_verified_cache('cloth-copurchase',cache,s['encoder_sha256'])
    if tuple(gpu.shape)!=(125839,1024) or not torch.isfinite(gpu).all():raise ValueError('Full finite1024-wide cache required')
    raw=ROOT/'data/raw/mm_graph/cloth-copurchase/lp-edge-split.pt'
    if fingerprint(raw)!=audit['raw_sha256']['lp-edge-split.pt']:raise ValueError('Official split changed')
    split=torch.load(str(raw),map_location='cpu',weights_only=False,mmap=True)
    for key,count in [('valid',121323),('test',97456)]:
        if np.asarray(split[key]['target_node_neg']).shape!=(count,150):raise ValueError('Complete official fixed150 negatives required')
    path=Path(read(run/'checkpoints/location.json')['checkpoint_directory'])/'link_head_best.pt'
    head=make_head(s['head_variant']).cuda();head.load_state_dict(torch.load(str(path),map_location='cpu',weights_only=False)['head']);head.eval()
    valid=split['valid'];indices={k:np.asarray(valid[k])[:64] for k in ['source_node','target_node','target_node_neg']}
    with use_cached_vectors(),torch.no_grad():
        a=link.vectors(gpu,indices['source_node']);b=link.vectors(gpu,indices['target_node'])
        c=link.vectors(gpu,indices['target_node_neg'].ravel()).reshape(64,150,-1)
        ranks=(1/ranking_metrics(head(a,b),head(a[:,None,:].expand_as(c),c))['mrr']).cpu().numpy()
    with np.load(run/'validation_link_ranks.npz') as values:np.testing.assert_array_equal(ranks,values['rank'][:64])
    config=copy.deepcopy(read(run/'config.json'));config['name']='cloth_final_runtime_check'
    config['evaluation'].update(final_profile_sha256=fingerprint(PROFILE),runtime_validation_rows=64,test_evaluations=0)
    e=Experiment(config,audit,path)
    try:
        result=dict(passed=True,actual_Torch_DGL_CUDA=True,full_cache_rows=125839,width=1024,
            fixed_head_validation64_rank_exact=True,reused_full_validation_and_optimizer_RNG_resume_acceptance=True,
            test_evaluations=0,selected_head_sha256=fingerprint(path),release_sha256=fingerprint(RELEASE))
        e.metric(dict(step=1,validation_probe_max_rank_difference=0.));atomic_write_json(e.directory/'summary.json',result);e.status('completed_cloth_final_runtime_check')
    finally:e.close()


def summarize(ledger,p):
    import numpy as np
    from came_repro.atomic_io import atomic_write_json
    rows=sorted(ledger['results'].values(),key=lambda r:r['seed']);values=np.array([r['score_pct'] for r in rows])
    stats=dict(completed=len(rows),expected=10,all_seeds_retained=True,metric_admission_gate=False,
        paper_pct=p['paper_mrr_pct'],paper_reproduction_complete=False,shared_pretraining_checkpoints=1)
    if len(rows):stats.update(mean_pct=float(values.mean()),sample_sd_pp=float(values.std(ddof=1)) if len(rows)>1 else None,
        gap_pp=float(values.mean()-p['paper_mrr_pct']),mean_within_two_pp=bool(abs(values.mean()-p['paper_mrr_pct'])<=2))
    atomic_write_json(OUT/'results.json',dict(state=ledger['state'],statistics=stats,results=rows))
    with (OUT/'results.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=['seed','score_pct','gap_pp','within_two_pp','official_test_queries','reverse_nonoverlap_mrr_pct','reused','run_directory'],extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4));ids=[r['seed'] for r in rows]
        ax.plot(ids,values,'o-',color='#315a7d',label='All completed seeds')
        ax.axhline(p['paper_mrr_pct'],ls='--',color='black',label='Paper28.88')
        ax.set(xticks=ids,xlabel='Downstream initialization seed',ylabel='Official test MRR x100',title='Cloth: %d/10 results'%len(rows));ax.legend(fontsize=8)
        fig.text(.02,.01,'Source: results.csv; no score admission or seed filtering; one frozen encoder; train-bidirected adaptation; reverse subset reported.',fontsize=7)
        fig.tight_layout(rect=(0,.07,1,1));fig.savefig(OUT/'cloth_final_comparison.png',dpi=180);plt.close(fig)


def pipeline(check_only=False):
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.pause_control import TrainingPaused
    from came_repro.storage_budget import check_budget
    from cloth_wn_autonomous import artifact_hashes
    from remaining_delivery import audit_test
    p=verify();OUT.mkdir(parents=True,exist_ok=True);guard=acquire_gpu_lock(ROOT/'.runtime/cloth_final_repeats.lock')
    ledger=read(LEDGER) if LEDGER.exists() else dict(release_sha256=fingerprint(RELEASE),state='ready',records={},results={},active_seconds=0.)
    logger=logging.getLogger('cloth_final_repeats');logger.setLevel(logging.INFO)
    handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'pipeline.log',encoding='utf-8')]
    for h in handlers:
        fmt=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s');fmt.converter=time.gmtime;h.setFormatter(fmt);logger.addHandler(h)
    def persist():atomic_write_json(LEDGER,ledger)
    sleep_guard=False
    try:
        if ledger['release_sha256']!=fingerprint(RELEASE):raise ValueError('Refuse changing an active/historical release')
        if PAUSE.exists():
            history=OUT/'pause_history';history.mkdir(exist_ok=True)
            atomic_write_json(history/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),read(PAUSE));PAUSE.unlink()
        if os.name=='nt':
            import ctypes
            sleep_guard=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        waiting=False
        while True:
            if PAUSE.exists():raise TrainingPaused('User requested final Cloth pause')
            try:probe=acquire_gpu_lock();probe.close();break
            except RuntimeError:
                if not waiting:logger.info('WAIT_GPU：等待原诊断保存完整轮退出，不启动并发GPU。');waiting=True
                time.sleep(10)
        check_budget(**p['budget'])
        def execute(name,args,filename=None):
            verify()
            if PAUSE.exists():raise TrainingPaused('User requested final Cloth pause')
            rec=ledger['records'].setdefault(name,dict(state='ready',attempts=[]))
            if rec['state']=='completed':
                for file,digest in rec['artifacts'].items():
                    if fingerprint(file)!=digest:raise ValueError('Completed result changed')
                return rec
            parent=best_resume(rec,filename) if filename else None
            if name.startswith('test_') and not parent:
                for attempt in rec['attempts']:
                    old=attempt.get('run_directory')
                    if old and (Path(old)/'test_link_ranks.npz').exists():raise ValueError('Saved test without consistent cursor requires audit')
            if parent:args+=['--resume-head',str(parent)]
            ledger['state']='running_'+name;persist();logger.info('JOB %s parent=%s',name,parent)
            code,directory=run_child(args,PAUSE,ledger,LEDGER,rec,logger,float('inf'))
            if code==75 or PAUSE.exists():raise TrainingPaused('Full epoch or official test cursor saved')
            if code or not directory:raise RuntimeError('Child did not complete: '+name+' exit='+str(code))
            rec.update(state='completed',run_directory=directory,summary=read(Path(directory)/'summary.json'),artifacts=artifact_hashes(directory));persist();return rec
        receipt=execute('runtime_check',[str(Path(__file__)),'--stage','check'])
        if not receipt['summary']['passed']:raise ValueError('Runtime CUDA acceptance failed')
        if check_only:ledger['state']='runtime_check_passed';persist();return ledger
        seed0=ROOT/p['seed0_confirmation_run'];zero=audit_test('cloth',seed0)
        zero.update(seed=0,run_directory=str(seed0),gap_pp=zero['score_pct']-28.88,within_two_pp=abs(zero['score_pct']-28.88)<=2,reused=True)
        if '0' in ledger['results'] and ledger['results']['0']!=zero:raise ValueError('Cannot replace original seed0')
        ledger['results']['0']=zero;persist();summarize(ledger,p)
        logger.info('CLOTH_RESULT 1/10 seed=0 MRR=%.8f%% reused; score does not gate training',zero['score_pct'])
        worker=str(ROOT/'scripts/remaining_delivery_job.py');selection=ROOT/p['selection_file']
        for seed in p['seeds'][1:]:
            key=str(seed)
            if key in ledger['results']:
                for file,digest in ledger['records']['test_s'+key]['artifacts'].items():
                    if fingerprint(file)!=digest:raise ValueError('Saved test result changed')
                continue
            logger.info('CLOTH %d/10 seed=%d fixed recipe; full training/validation/test; no score gate',seed+1,seed)
            training=execute('train_s'+key,[worker,'--stage','train','--task','cloth','--seed',key,'--selection',str(selection)],'link_head_last.pt')
            run=Path(training['run_directory']);s=training['summary'];cache=Path(s['embedding_cache'])
            head=Path(read(run/'checkpoints/location.json')['checkpoint_directory'])/'link_head_best.pt'
            lock=dict(dataset='cloth-copurchase',encoder_sha256=s['encoder_sha256'],classifier_checkpoint=str(head),classifier_sha256=fingerprint(head),
                validation_run_directory=str(run),validation_summary_sha256=fingerprint(run/'summary.json'),embedding_cache=str(cache),
                inference_state_sha256=fingerprint(cache/'inference_state.json'),locked_before_test=True,test_used_for_selection=False,
                graph_policy=s['graph_policy'],head_variant=s['head_variant'],selected_validation_mrr=s['validation_mrr'])
            file=OUT/('seed'+key+'_test_lock.json')
            if file.exists() and read(file)!=lock:raise ValueError('Frozen validation-selected test head changed')
            atomic_write_json(file,lock)
            tested=execute('test_s'+key,[worker,'--stage','test','--task','cloth','--selection',str(file)],'test_cursor.json')
            result=audit_test('cloth',tested['run_directory'])
            if result['official_test_queries']!=97456:raise ValueError('Complete official test required')
            result.update(seed=seed,run_directory=tested['run_directory'],gap_pp=result['score_pct']-28.88,
                within_two_pp=abs(result['score_pct']-28.88)<=2,reused=False)
            ledger['results'][key]=result;persist();summarize(ledger,p)
            logger.info('CLOTH_RESULT %d/10 seed=%d MRR=%.8f%% gap=%+.8fpp; retained',len(ledger['results']),seed,result['score_pct'],result['gap_pp'])
        ledger['state']='completed_ten_seeds';persist();summarize(ledger,p);return ledger
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dry-run',action='store_true');parser.add_argument('--check-only',action='store_true');parser.add_argument('--pause',action='store_true');parser.add_argument('--detach',action='store_true');parser.add_argument('--stage',choices=['check'])
    args=parser.parse_args()
    for stream in [sys.stdout,sys.stderr]:
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    if args.pause:
        atomic_write_json(PAUSE,dict(reason='user_requested_final_cloth_pause'));print('已请求完整轮/测试游标边界暂停；下次点同文件继续。');return
    verify()
    if args.stage:check_gpu();return
    print('Cloth作业08最终重复：复用seed0，完成seed1..9。指标差距只报告，不再阻止训练。',flush=True)
    print('固定最高完整验证方案；官方全数据/150负例/1024/CUDA；不删除或挑seed。',flush=True)
    print('日志/结果：'+str(OUT),flush=True)
    if args.dry_run:print('配置/源码/已选权重/既有恢复验收核对通过；GPU作业0。');return
    for name in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']:os.environ[name]='1'
    if not args.detach:pipeline(args.check_only);return
    import psutil,subprocess
    OUT.mkdir(parents=True,exist_ok=True);guard=acquire_gpu_lock(ROOT/'.runtime/cloth_final_repeats_launch.lock')
    try:
        launch=OUT/'launch.json'
        if launch.exists():
            old=read(launch)
            try:
                p=psutil.Process(old['pid'])
                if abs(p.create_time()-old['creation_time'])<1:print('本入口已运行，未启动重复任务。');return
            except psutil.NoSuchProcess:pass
        log=OUT/('console_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.log')
        arguments=[sys.executable,'-u',str(Path(__file__))]
        if args.check_only:arguments.append('--check-only')
        with log.open('ab') as output:
            child=subprocess.Popen(arguments,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,close_fds=True,
                creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB,
                env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'})
        atomic_write_json(launch,dict(pid=child.pid,creation_time=psutil.Process(child.pid).create_time(),console_log=str(log),one_shot_not_recurring=True));print('已启动离线Cloth最终重复，PID='+str(child.pid))
    finally:guard.close()


if __name__=='__main__':main()
