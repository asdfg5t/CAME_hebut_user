"""One frozen Cloth recipe; test gate controls repetition cost, never recipe search."""
import copy
import csv
import datetime
import json
import logging
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
OUT = ROOT/'artifacts/cloth_target_delivery'
PROFILE = ROOT/'configs/cloth_target_delivery_20261018.json'
RELEASE = ROOT/'artifacts/delivery/cloth_target_release.json'
LEDGER = ROOT/'artifacts/workflow_state/cloth_target_delivery_current.json'
PAUSE = ROOT/'artifacts/workflow_state/cloth_target_delivery_pause.json'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_profile(p):
    required = dict(dataset='cloth-copurchase', nodes=125839, width=1024,
        official_validation_queries=121323, official_test_queries=97456, fixed_negative_candidates=150,
        target_absolute_gap_pp=2., maximum_new_seed0_confirmations=1, training_kind='dropout_bce')
    if any(p.get(k) != v for k,v in required.items()) or p.get('seeds') != list(range(10)):
        raise ValueError('Cloth必须固定完整数据、全部seed及一次确认协议。')
    if not p['no_automatic_strategy_search_after_test'] or not p['prior_diagnostic_half_pp_gate_not_reclassified']:
        raise ValueError('禁止测试后自动搜策略或重写历史验证门槛。')


def within_target(score_pct, p):
    if not math.isfinite(score_pct): raise ValueError('Nonfinite Cloth score')
    return abs(score_pct-p['paper_mrr_pct']) <= p['target_absolute_gap_pp']


def verify():
    from came_repro.author_loader import fingerprint, verify_originals
    from remaining_delivery import verify_release
    verify_release();verify_originals();p=read(PROFILE);validate_profile(p)
    release=read(RELEASE)
    for name,digest in release['files'].items():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        if fingerprint(path)!=digest: raise ValueError('Cloth冻结输入已改变：'+name)
    run=ROOT/p['validation_run'];s=read(run/'summary.json')
    if (fingerprint(run/'summary.json')!=p['validation_summary_sha256'] or s['seed']!=0
            or s['test_evaluations']!=0 or not read(run/'status.json')['status'].startswith('completed')):
        raise ValueError('必须使用锁定的完整seed0验证候选，不能替换或重复选模型。')
    return p


def freeze_selection(p):
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    run=ROOT/p['validation_run'];s=read(run/'summary.json')
    recipe={k:s[k] for k in ['head_variant','lr','batch_size','max_epochs','patience',
        'step_lr_size','step_lr_gamma','loss_reduction','head_training_dropout']}
    value=dict(release_sha256=fingerprint(RELEASE), tasks=dict(cloth=dict(
        validation_run=str(run), dataset=p['dataset'], embedding_cache=s['embedding_cache'],
        graph_policy=s['graph_policy'], training_kind='dropout_bce', recipe=recipe)),
        selection_basis=p['selection'], historical_tests_seen=True, test_used_for_selection=False)
    path=OUT/'protocol_selection.json'
    if path.exists() and read(path)!=value: raise ValueError('已锁定训练方案不得改变。')
    atomic_write_json(path,value);return path,value


def test_lock(run,p):
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    run=Path(run);s=read(run/'summary.json');cache=Path(s['embedding_cache'])
    head=Path(read(run/'checkpoints/location.json')['checkpoint_directory'])/'link_head_best.pt'
    lock=dict(dataset=p['dataset'], encoder_sha256=p['encoder_sha256'], classifier_checkpoint=str(head),
        classifier_sha256=fingerprint(head), validation_run_directory=str(run),
        validation_summary_sha256=fingerprint(run/'summary.json'), embedding_cache=str(cache),
        inference_state_sha256=fingerprint(cache/'inference_state.json'), locked_before_test=True,
        test_used_for_selection=False, graph_policy=p['graph_policy'], head_variant=s['head_variant'],
        selected_validation_mrr=s['validation_mrr'])
    path=OUT/('seed'+str(s['seed'])+'_test_lock.json')
    if path.exists() and read(path)!=lock: raise ValueError('已锁定测试头不得替换。')
    atomic_write_json(path,lock);return path


def preflight():
    import numpy as np
    import torch
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.experiment import Experiment, seed_everything
    from came_repro.lp_validation_repair import load_verified_cache, make_head
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro import pause_control as pauses, link_evaluation as link
    from came_repro.cloth_regularized_train import train_regularized
    from cloth_wn_autonomous_backend import assert_same
    from remaining_delivery_job import encoder, resources
    p=verify();reference=ROOT/p['validation_run'];s=read(reference/'summary.json')
    selection,item=freeze_selection(p);recipe=item['tasks']['cloth']['recipe']
    seed_everything(731);vectors=torch.randn(16,1024,device='cuda')
    train=dict(source_node=np.arange(6),target_node=np.arange(6)+1)
    valid=dict(source_node=np.array([8,9]),target_node=np.array([9,10]),target_node_neg=np.tile(np.arange(16),(2,10))[:,:150])
    fixture_recipe=dict(recipe,batch_size=3,max_epochs=3)
    config=dict(name='cloth_gate_resume_fixture',seed=1,storage=resources(),evaluation=dict(fixture_only_not_benchmark=True))
    runs=[]
    original_pause=pauses.pause_requested
    for mode in ['continuous','paused','resumed']:
        parent=runs[1].checkpoint_directory/'link_head_last.pt' if mode=='resumed' else None
        e=Experiment(copy.deepcopy(config),dict(fixture_only_not_benchmark=True),parent);runs.append(e)
        calls=[0]
        def pause_two(): calls[0]+=1;return calls[0]==2
        if mode=='paused': pauses.pause_requested=pause_two
        try:
            try:
                with use_cached_vectors(): train_regularized(vectors,train,valid,fixture_recipe,1,e,parent)
            except pauses.TrainingPaused:
                if mode!='paused': raise
                e.status('paused_fixture')
            else:
                if mode=='paused': raise ValueError('Resume fixture did not pause')
                e.status('completed_fixture')
        finally: pauses.pause_requested=original_pause;e.close()
    a,b=[torch.load(str(runs[i].checkpoint_directory/'link_head_last.pt'),map_location='cpu',weights_only=False) for i in [0,2]]
    for key in ['head','optimizer','scheduler','rng','sampling_generator','history_rows','best_state','selected_epoch','next_epoch']:
        assert_same(a[key],b[key])
    if a['next_epoch']!=4: raise ValueError('Resume fixture incomplete')
    del a,b,vectors;torch.cuda.empty_cache()
    gpu,audit=load_verified_cache(p['dataset'],Path(s['embedding_cache']),p['encoder_sha256'])
    if tuple(gpu.shape)!=(125839,1024) or audit['diagnostic_topology']['policy']!=p['graph_policy']:
        raise ValueError('Cloth complete cache/topology changed')
    raw=ROOT/'data/raw/mm_graph/cloth-copurchase/lp-edge-split.pt'
    if fingerprint(raw)!=audit['raw_sha256']['lp-edge-split.pt']: raise ValueError('Official split changed')
    split=torch.load(str(raw),map_location='cpu',weights_only=False,mmap=True)
    for key,count in [('valid',121323),('test',97456)]:
        if np.asarray(split[key]['target_node_neg']).shape!=(count,150): raise ValueError('Full official fixed negative candidates required')
    head=make_head(s['head_variant']).cuda();directory=Path(read(reference/'checkpoints/location.json')['checkpoint_directory'])
    head.load_state_dict(torch.load(str(directory/'link_head_best.pt'),map_location='cpu',weights_only=False)['head']);head.eval()
    e=Experiment(dict(name='cloth_gate_full_validation',seed=0,storage=resources(),evaluation=dict(
        release_sha256=fingerprint(RELEASE),test_used_for_selection=False,head_frozen=True)),audit,encoder())
    try:
        with use_cached_vectors(),torch.no_grad():
            value=link.evaluate_split(head,gpu,split['valid'],64,np.zeros(121323,dtype=bool),e.directory/'validation_link_ranks.npz')
        measured=value['official']['mrr']
        if abs(measured-s['validation_mrr'])>1e-12: raise ValueError('Frozen full official validation replay changed')
        result=dict(passed=True,actual_Torch_DGL_CUDA=True,validation_mrr=measured,official_validation_queries=121323,
            full_optimizer_RNG_sampler_history_resume_verified=True,fixture_runs=[str(r.directory) for r in runs],
            test_evaluations=0,release_sha256=fingerprint(RELEASE),source_selection_sha256=fingerprint(selection))
        e.metric(dict(step=1,validation_mrr=measured));atomic_write_json(e.directory/'summary.json',result);e.status('completed_cloth_gate_preflight')
    except BaseException as error: e.status('failed',error=str(error));raise
    finally: e.close()


def worker(args):
    from came_repro.protocol_repair import check_cuda
    import torch
    from came_repro.experiment import seed_everything
    verify();check_cuda();torch.set_num_threads(4);seed_everything(args.seed)
    if args.stage=='preflight': preflight()
    elif args.stage=='train':
        from remaining_delivery_job import train_task
        train_task('cloth',args.seed,args.selection,args.resume_head)
    else:
        from remaining_delivery_test import confirm
        confirm('cloth',args.selection,args.resume_head)


def summarize(ledger,p):
    import numpy as np
    from came_repro.atomic_io import atomic_write_json
    rows=sorted(ledger['results'].values(),key=lambda r:r['seed']);values=np.array([r['score_pct'] for r in rows])
    stats=dict(completed=len(rows),expected=10,mean_pct=float(values.mean()) if len(rows) else None,
        sample_sd_pp=float(values.std(ddof=1)) if len(rows)>1 else None,all_seeds_retained=True,
        paper_pct=p['paper_mrr_pct'],target_absolute_gap_pp=2.,admission_gate=ledger.get('admission'),
        conditional_repetition_requested_by_user=True,paper_reproduction_complete=False)
    if len(rows): stats.update(gap_pp=float(values.mean()-p['paper_mrr_pct']),mean_within_two_pp=within_target(float(values.mean()),p))
    atomic_write_json(OUT/'results.json',dict(state=ledger['state'],statistics=stats,results=rows))
    with (OUT/'results.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=['seed','score_pct','gap_pp','within_two_pp','official_test_queries','reverse_nonoverlap_mrr_pct','run_directory'],extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4));ax.plot([r['seed'] for r in rows],values,'o-',color='#315a7d',label='All completed seeds')
        ax.axhline(p['paper_mrr_pct'],color='black',ls='--',label='Paper 28.88')
        ax.axhspan(26.88,30.88,color='#dce6e8',alpha=.6,label='Absolute gap <=2 pp')
        ax.set(xlabel='Downstream initialization seed',ylabel='Official test MRR x100',title='Cloth: %d/10 results'%len(rows));ax.legend(fontsize=8)
        fig.text(.02,.01,'Source: results.csv; one shared frozen encoder; bidirected train adaptation; reverse-overlap subset reported separately.',fontsize=7)
        fig.tight_layout(rect=(0,.07,1,1));fig.savefig(OUT/'cloth_test_comparison.png',dpi=180);plt.close(fig)


def run(check_only=False):
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.pause_control import TrainingPaused
    from came_repro.storage_budget import check_budget
    from cloth_wn_autonomous import artifact_hashes
    from remaining_delivery import audit_test
    p=verify();OUT.mkdir(parents=True,exist_ok=True)
    guard=acquire_gpu_lock(ROOT/'.runtime/cloth_target_delivery.lock')
    ledger=read(LEDGER) if LEDGER.exists() else dict(release_sha256=fingerprint(RELEASE),state='ready',records={},results={},active_seconds=0.)
    logger=logging.getLogger('cloth_target');logger.setLevel(logging.INFO)
    handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'pipeline.log',encoding='utf-8')]
    for h in handlers:
        fmt=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s');fmt.converter=time.gmtime;h.setFormatter(fmt);logger.addHandler(h)
    def persist(): atomic_write_json(LEDGER,ledger)
    sleep_guard=False
    try:
        if ledger['release_sha256']!=fingerprint(RELEASE): raise ValueError('另一交付身份，禁止覆盖历史。')
        if PAUSE.exists():
            history=OUT/'pause_history';history.mkdir(exist_ok=True)
            atomic_write_json(history/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),read(PAUSE));PAUSE.unlink()
        if sys.platform=='win32':
            import ctypes
            sleep_guard=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        waiting=False
        while True:
            if PAUSE.exists(): raise TrainingPaused('用户暂停Cloth。')
            try: probe=acquire_gpu_lock();probe.close();break
            except RuntimeError:
                if not waiting: logger.info('WAIT_GPU：保留现有任务，等待锁释放后串行检测。');waiting=True
                time.sleep(10)
        check_budget(**p['budget']);selection,_=freeze_selection(p)
        entry=str(ROOT/'scripts/run_15_cloth_target.py')
        def execute(name,args,checkpoint=None):
            verify()
            if PAUSE.exists(): raise TrainingPaused('用户暂停Cloth。')
            record=ledger['records'].setdefault(name,dict(state='ready',attempts=[]))
            if record['state']=='completed':
                for path,digest in record['artifacts'].items():
                    if fingerprint(path)!=digest: raise ValueError('已完成产物改变：'+path)
                return record
            parent=best_resume(record,checkpoint) if checkpoint else None
            if name.startswith('test_') and not parent:
                for attempt in record['attempts']:
                    prior=attempt.get('run_directory')
                    if prior and (Path(prior)/'test_link_ranks.npz').exists(): raise ValueError('已有测试输出但游标丢失；需审计，不能自动重测。')
            if parent: args+=['--resume-head',str(parent)]
            ledger['state']='running_'+name;persist();logger.info('JOB %s parent=%s',name,parent)
            code,run_dir=run_child([entry,*args],PAUSE,ledger,LEDGER,record,logger,float('inf'))
            if code==75 or PAUSE.exists(): raise TrainingPaused('已保存完整轮/测试游标。')
            if code or not run_dir: raise RuntimeError('Cloth子任务失败：%s exit=%s'%(name,code))
            record.update(state='completed',run_directory=run_dir,summary=read(Path(run_dir)/'summary.json'),artifacts=artifact_hashes(run_dir));persist();return record
        accepted=execute('preflight',['--stage','preflight'])
        if not accepted['summary']['passed']: raise ValueError('真实CUDA/恢复/全验证验收未通过。')
        if check_only: ledger['state']='preflight_passed';persist();return ledger
        for seed in range(10):
            key=str(seed)
            if key in ledger['results']:
                prior=ledger['records']['test_s'+key]
                for path,digest in prior['artifacts'].items():
                    if fingerprint(path)!=digest: raise ValueError('已保存的测试产物改变。')
            else:
                run_dir=ROOT/p['validation_run'] if seed==0 else Path(execute('train_s'+key,
                    ['--stage','train','--seed',key,'--selection',str(selection)],'link_head_last.pt')['run_directory'])
                lock=test_lock(run_dir,p)
                tested=execute('test_s'+key,['--stage','test','--selection',str(lock)],'test_cursor.json')
                result=audit_test('cloth',tested['run_directory'])
                if result['official_test_queries']!=97456: raise ValueError('不完整官方测试。')
                result.update(seed=seed,run_directory=tested['run_directory'],gap_pp=result['score_pct']-28.88,
                    within_two_pp=within_target(result['score_pct'],p),seed0_training_reused=seed==0)
                ledger['results'][key]=result;persist();summarize(ledger,p)
                logger.info('RESULT seed=%d MRR=%.8f%% gap=%+.8fpp',seed,result['score_pct'],result['gap_pp'])
            if seed==0:
                passed=within_target(ledger['results']['0']['score_pct'],p)
                ledger['admission']=dict(passed=passed,seed0_score_pct=ledger['results']['0']['score_pct'],
                    used_only_to_authorize_repetition_cost=True,recipe_changed_after_test=False)
                if not passed:
                    ledger['state']='stopped_target_not_met_no_strategy_search';persist();summarize(ledger,p);return ledger
        ledger['state']='completed_ten_seeds';persist();summarize(ledger,p);return ledger
    except BaseException as error:
        ledger.update(state='paused' if isinstance(error,(TrainingPaused,KeyboardInterrupt)) or PAUSE.exists() else 'failed',error=str(error));persist();raise
    finally:
        if sleep_guard:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for h in handlers: h.close();logger.removeHandler(h)
        guard.close()
