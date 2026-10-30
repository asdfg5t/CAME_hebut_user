"""Goodreads-LP: real validation-first check, official full test, resumable ten seeds."""
import argparse
import copy
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
if __name__=='__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
OUT=ROOT/'artifacts/goodreads_official_rebuild'
PROFILE=ROOT/'configs/goodreads_official_rebuild_20261021.json'
RELEASE=ROOT/'artifacts/delivery/goodreads_official_rebuild_release.json'
LEDGER=ROOT/'artifacts/workflow_state/goodreads_official_rebuild_current.json'
PAUSE=ROOT/'artifacts/workflow_state/goodreads_official_rebuild_pause.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify():
    from came_repro.author_loader import fingerprint,verify_originals
    verify_originals();release=read(RELEASE)
    for name,digest in release['input_sha256'].items():
        if fingerprint(Path(name))!=digest:raise ValueError('Frozen Goodreads input changed: '+name)
    p=read(PROFILE);r=p['recipe']
    if fingerprint(Path(p['encoder_checkpoint']))!=p['encoder_sha256']:raise ValueError('Frozen original encoder changed')
    if (p['dataset']!='books-lp' or p['seeds']!=list(range(10)) or p['test_score_admission_gate'] is not False
        or p['official_test_queries']!=1710677 or p['fixed_test_negatives']!=150 or p['cache_width']!=1024
        or r!={'lr':.0005,'batch_size':512,'positive_budget_per_round':512000,'max_rounds':30,'patience':10,'scheduler_step':5,'scheduler_gamma':.1}):
        raise ValueError('Scientific Goodreads protocol changed')
    return p


def config_for(p,seed):
    config=read(ROOT/p['reference_run']/'config.json')
    config.update(name='goodreads_official_head',seed=seed,storage=read(ROOT/'configs/storage.json'))
    config['evaluation']=dict(dataset='books-lp',seed=seed,embedding_cache=p['embedding_cache'],
        recipe=p['recipe'],test_used_for_selection=False,release_sha256=__import__('came_repro.author_loader',fromlist=['fingerprint']).fingerprint(RELEASE))
    config['deviations']=list(config.get('deviations',[]))+[
        'LP head and optimizer recipe reconstructed from pinned MM-Graph, not confirmed CAME author training code.',
        'Three-linear ReLU Hadamard predictor retains1024-wide hidden layers rather than upstream default256.',
        'At most512000 positives per optimization round, sampled from the full official3437017-edge pool; NOT full official epochs.',
        'Uniform(1) train target draws; concatenated BCE mean; Adam5e-4 StepLR5gamma0.1; max30 rounds; patience10 is a declared reconstruction.',
        'Complete official150 validation/test negatives unchanged; no filtered scoring, no target-score parameter search.']
    return config


def acceptance(p):
    import numpy as np
    import torch
    import zipfile
    from came_repro import goodreads_official as core
    from came_repro.protocol_repair import check_cuda
    from came_repro.experiment import Experiment,seed_everything
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.pause_control import TrainingPaused
    from cloth_wn_autonomous_backend import assert_same
    check_cuda();torch.set_num_threads(4);seed_everything(713)
    finite_fixture=torch.ones(9,1024,device='cuda');core.require_finite_rows(finite_fixture,4)
    for bad_row in [0,8]:
        finite_fixture.fill_(1);finite_fixture[bad_row,-1]=float('nan')
        try:core.require_finite_rows(finite_fixture,4)
        except FloatingPointError:pass
        else:raise ValueError('CUDA finite check skipped a first/tail row')
    del finite_fixture
    archive=ROOT/p['upstream_archive']
    if fingerprint(archive)!=p['upstream_archive_sha256']:raise ValueError('Pinned upstream changed')
    with zipfile.ZipFile(archive) as z:source=z.read(p['upstream_model_member']).decode('utf-8')
    namespace={};exec(compile(source,p['upstream_model_member'],'exec'),namespace)
    upstream=namespace['SAGE'](1024,1024,2).cuda();candidate=core.make_head().cuda()
    upstream.predictor.load_state_dict(candidate.state_dict())
    x=torch.randn(7,1024,device='cuda');y=torch.randn_like(x)
    with torch.no_grad():torch.testing.assert_close(core.pair_logits(candidate,x,y),upstream.predictor(x*y).squeeze(-1),rtol=0,atol=0)
    del upstream,candidate,x,y
    gpu=torch.randn(24,1024,device='cuda')
    train=dict(source_node=np.arange(24,dtype=np.int64),target_node=np.roll(np.arange(24,dtype=np.int64),1))
    valid=dict(source_node=np.arange(5,dtype=np.int64),target_node=np.arange(1,6,dtype=np.int64),target_node_neg=np.tile(np.arange(150,dtype=np.int64)%24,(5,1)))
    recipe=dict(p['recipe'],positive_budget_per_round=16,batch_size=4,max_rounds=3)
    config=dict(name='goodreads_cuda_fixture',seed=1,storage=read(ROOT/'configs/storage.json'),evaluation=dict(dataset='synthetic_fixture_not_paper',recipe=recipe))
    runs=[]
    for mode in ['continuous','paused','resumed']:
        parent=runs[1].checkpoint_directory/'link_head_last.pt' if mode=='resumed' else None
        e=Experiment(config,dict(fixture_only=True),parent);runs.append(e)
        try:
            s=core.train_rounds(gpu,train,valid,recipe,1,e,parent,pause_hook=(lambda r,c,phase:r==2 and c==8 and phase=='train') if mode=='paused' else None)
            if mode=='paused':raise ValueError('Mid-round pause failed')
            atomic_write_json(e.directory/'summary.json',s);e.status('completed_fixture')
        except TrainingPaused:
            if mode!='paused':raise
            e.status('paused_fixture')
        finally:e.close()
    a,b=[torch.load(str(runs[i].checkpoint_directory/'link_head_last.pt'),map_location='cpu',weights_only=False) for i in [0,2]]
    keys=['head','optimizer','scheduler','rng','sampling_generator','next_round','positive_cursor','round_order','loss_sum','best_mrr','selected_round','stale','best_state','history_rows']
    for k in keys:assert_same(a[k],b[k])
    if a['next_round']!=4:raise ValueError('Incomplete resume acceptance')
    positive=torch.tensor([1.,1.,1.],device='cuda');negative=torch.stack([torch.ones(150,device='cuda'),torch.zeros(150,device='cuda'),torch.full((150,),2.,device='cuda')])
    np.testing.assert_array_equal(core.official_ranks(positive,negative).cpu().numpy(),[76.,1.,151.])
    del a,b,gpu;torch.cuda.empty_cache()
    gpu,split,audit=core.load_data(p)
    report=dict(passed=True,actual_torch_and_dgl_cuda=True,pinned_predictor_bitexact=True,mid_round_resume_exact_keys=keys,
        full_cache_finite_check_is_bounded_and_complete=True,CUDA_first_and_tail_NaN_rejection_verified=True,
        fixture_runs=[str(e.directory) for e in runs],official_nodes=len(gpu),official_valid_queries=len(split['valid']['source_node']),official_test_queries=len(split['test']['source_node']),official_fixed_negatives=150,test_evaluations=0)
    e=Experiment(dict(name='goodreads_actual_preflight',seed=0,storage=read(ROOT/'configs/storage.json'),evaluation=dict(dataset='books-lp',acceptance_only=True)),audit,Path(p['encoder_checkpoint']))
    try:
        atomic_write_json(e.directory/'summary.json',report);e.metric(dict(step=1,passed=1,official_test_queries=1710677,test_evaluations=0));e.status('completed_goodreads_preflight')
    finally:e.close()


def lock_head(run):
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    run=Path(run);s=read(run/'summary.json');head=Path(read(run/'checkpoints/location.json')['checkpoint_directory'])/'link_head_best.pt'
    if s['test_evaluations']!=0:raise ValueError('Validation-only run required')
    path=OUT/('seed%d_selection_lock.json'%s['seed'])
    value=dict(validation_run_directory=str(run),validation_summary_sha256=fingerprint(run/'summary.json'),classifier_checkpoint=str(head),classifier_sha256=fingerprint(head),locked_before_test=True,test_used_for_selection=False,selected_validation_mrr=s['validation_mrr'])
    if path.exists() and read(path)!=value:raise ValueError('Refuse to replace a frozen test selection')
    atomic_write_json(path,value);return path


def artifacts(run):
    from came_repro.author_loader import fingerprint
    run=Path(run);paths=[run/x for x in ['summary.json','status.json','config.json','manifest.json','logs/metrics.jsonl','logs/metrics.csv']]
    paths+=list(run.glob('*.npz'))+list((run/'figures').glob('*.png'))+list((run/'tensorboard').glob('*'))
    location=run/'checkpoints/location.json'
    if location.exists():paths+=[location];paths+=list(Path(read(location)['checkpoint_directory']).glob('*.pt'))
    return {str(x):fingerprint(x) for x in paths if x.is_file()}


def summarize(ledger):
    import numpy as np
    import csv
    from came_repro.atomic_io import atomic_write_json
    rows=sorted(ledger['results'].values(),key=lambda r:r['seed']);scores=[r['score_pct'] for r in rows]
    stats=dict(completed=len(rows),expected=10,paper_pct=10.42,all_seeds_retained=True,metric_admission_gate=False,paper_reproduction_complete=False)
    if rows:stats.update(mean_pct=float(np.mean(scores)),sample_sd_pp=float(np.std(scores,ddof=1)) if len(rows)>1 else None,gap_pp=float(np.mean(scores)-10.42))
    atomic_write_json(OUT/'results.json',dict(state=ledger['state'],statistics=stats,results=rows))
    if not rows:return
    with (OUT/'results.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));ax.plot([r['seed'] for r in rows],scores,'o-',label='All completed official test seeds');ax.axhline(10.42,color='black',ls='--',label='Paper10.42')
    ax.set(xlabel='Downstream seed',ylabel='Complete official test MRR (%)',title='Goodreads-LP: %d/10'%len(rows),xticks=[r['seed'] for r in rows]);ax.legend(fontsize=8)
    fig.text(.02,.01,'1710677 official queries; fixed150 negatives; pinned predictor reconstruction; no filtering or seed rejection.',fontsize=7)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(OUT/'goodreads_comparison.png',dpi=180);plt.close(fig)


def pipeline(prepare_only=False):
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.pause_control import TrainingPaused
    from came_repro.storage_budget import check_budget
    from remaining_delivery import audit_test
    p=verify();OUT.mkdir(parents=True,exist_ok=True);guard=acquire_gpu_lock(ROOT/'.runtime/goodreads_official.lock')
    logger=logging.getLogger('goodreads_official_pipeline');logger.setLevel(logging.INFO);handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'pipeline.log',encoding='utf-8')]
    for h in handlers:
        formatter=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s');formatter.converter=time.gmtime;h.setFormatter(formatter);logger.addHandler(h)
    ledger=read(LEDGER) if LEDGER.exists() else dict(state='ready',release_sha256=fingerprint(RELEASE),active_seconds=0.,records={},results={})
    def persist():atomic_write_json(LEDGER,ledger)
    sleep_guard=False
    try:
        if ledger['release_sha256']!=fingerprint(RELEASE):raise ValueError('Existing frozen experiment cannot change identity')
        if PAUSE.exists():
            history=OUT/'pause_history';history.mkdir(exist_ok=True);atomic_write_json(history/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),read(PAUSE));PAUSE.unlink()
        probe=acquire_gpu_lock();probe.close()
        from came_repro.goodreads_storage import reserve
        reserve(check_budget(additional_project_bytes=256*2**20,additional_cache_bytes=2**30))
        if os.name=='nt':
            import ctypes
            sleep_guard=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        def execute(name,stage,seed=0,lock=None):
            verify()
            rec=ledger['records'].setdefault(name,dict(state='ready',attempts=[]))
            if rec['state']=='completed':
                for file,sha in rec['artifacts'].items():
                    if fingerprint(file)!=sha:raise ValueError('Completed artifact changed')
                return rec
            if PAUSE.exists():raise TrainingPaused('User paused Goodreads')
            parent=best_resume(rec,'link_head_last.pt' if stage=='train' else 'test_cursor.json') if stage!='check' else None
            args=[str(Path(__file__)),'--stage',stage,'--seed',str(seed)]
            if lock:args+=['--selection-lock',str(lock)]
            if parent:args+=['--resume-head',str(parent)]
            ledger['state']='running_'+name;persist();logger.info('JOB %s parent=%s',name,parent)
            code,run=run_child(args,PAUSE,ledger,LEDGER,rec,logger,float('inf'))
            if code==75 or PAUSE.exists():raise TrainingPaused('Complete Goodreads state saved')
            if code or not run:raise RuntimeError('Goodreads stage failed: %s exit=%s'%(name,code))
            rec.update(state='completed',run_directory=run,summary=read(Path(run)/'summary.json'),artifacts=artifacts(run));persist();return rec
        checked=execute('runtime_check','check')
        if not checked['summary']['passed']:raise ValueError('Actual CUDA/resume acceptance required')
        for seed in range(10):
            key=str(seed)
            if key not in ledger['results']:
                trained=execute('train_s'+key,'train',seed);lock=lock_head(trained['run_directory'])
                tested=execute('test_s'+key,'test',seed,lock);result=audit_test('books',tested['run_directory'])
                result.update(seed=seed,run_directory=tested['run_directory'],paper_pct=10.42,gap_pp=result['score_pct']-10.42)
                ledger['results'][key]=result;persist();logger.info('GOODREADS_RESULT %d/10 seed=%d official_MRR=%.8f%% gap=%+.8fpp retained',len(ledger['results']),seed,result['score_pct'],result['gap_pp'])
            else:
                for file,sha in ledger['records']['test_s'+key]['artifacts'].items():
                    if fingerprint(file)!=sha:raise ValueError('Completed result changed')
            if prepare_only and seed==0:
                ledger['state']='ready_for_user_repeats' if len(ledger['results'])<10 else 'completed_ten_seeds';persist();summarize(ledger);return
            summarize(ledger)
        ledger['state']='completed_ten_seeds';persist();summarize(ledger)
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dry-run',action='store_true');parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--pause',action='store_true');parser.add_argument('--stage',choices=['check','train','test']);parser.add_argument('--seed',type=int,default=0);parser.add_argument('--resume-head',type=Path);parser.add_argument('--selection-lock',type=Path);args=parser.parse_args()
    for stream in [sys.stdout,sys.stderr]:
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    if args.pause:atomic_write_json(PAUSE,dict(reason='user_requested_Goodreads_pause'));print('已请求保存训练batch/测试query断点；下次运行同文件继续。');return
    p=verify()
    if args.seed not in range(10):raise ValueError('Only fixed seeds0..9 allowed')
    if args.dry_run:print('冻结身份核验通过，真实GPU任务0；不是训练/测试完成。');return
    if args.stage:
        from came_repro.protocol_repair import check_cuda
        from came_repro.experiment import Experiment
        from came_repro import goodreads_official as core
        import torch
        check_cuda();torch.set_num_threads(4)
        if args.stage=='check':acceptance(p);return
        if args.stage=='test':core.test_saved(p,args.selection_lock,args.resume_head);return
        if args.resume_head:
            from came_repro.author_loader import fingerprint
            previous=torch.load(str(args.resume_head),map_location='cpu',weights_only=False)
            if previous['config']['evaluation']['release_sha256']!=fingerprint(RELEASE):
                migration=read(ROOT/'artifacts/audit/goodreads_finite_check_migration_20261021.json')
                if fingerprint(args.resume_head)!=migration['parent_checkpoint_sha256']:raise ValueError('Unapproved source-version checkpoint migration')
            del previous
        gpu,split,audit=core.load_data(p);config=config_for(p,args.seed);e=Experiment(config,audit,args.resume_head or Path(p['encoder_checkpoint']))
        try:
            s=core.train_rounds(gpu,split['train'],split['valid'],p['recipe'],args.seed,e,args.resume_head)
            s.update(encoder_sha256=p['encoder_sha256'],embedding_cache=p['embedding_cache']);atomic_write_json(e.directory/'summary.json',s);e.status('completed_goodreads_validation')
        except BaseException as error:
            from came_repro.pause_control import TrainingPaused
            e.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
        finally:e.close()
        return
    print('Goodreads-LP：真实CUDA/一致断点验收→seed0完整实测→固定十seed；无分数准入，无模拟结果。',flush=True)
    print('日志/统计：'+str(OUT),flush=True);pipeline(args.prepare_only)


if __name__=='__main__':main()
