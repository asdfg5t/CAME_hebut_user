"""Frozen one-entry delivery for main tasks06/08/09/10; all seeds are retained."""
import csv
import datetime
import json
import logging
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
OUT=ROOT/'artifacts/remaining_tasks_delivery'
PROFILE=ROOT/'configs/remaining_tasks_delivery_20261018.json'
RELEASE=ROOT/'artifacts/delivery/remaining_tasks_release.json'
LEDGER=ROOT/'artifacts/workflow_state/remaining_tasks_delivery_current.json'
PAUSE=ROOT/'artifacts/workflow_state/remaining_tasks_delivery_pause.json'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def validate_profile(profile):
    if (profile['seeds']!=list(range(10)) or set(profile['tasks'])!={'wn','cloth','books','products'}
            or profile['capacity']['nodes']!=2449029 or profile['capacity']['width']!=1024
            or profile['capacity']['official_test_nodes']!=2213091 or profile['capacity']['minimum_nodes_per_second']!=30):
        raise ValueError('交付协议必须保留四任务、全部固定seed、完整数据和数值/资源门槛。')


def verify_release():
    from came_repro.author_loader import fingerprint,verify_originals
    verify_originals();release=read(RELEASE);validate_profile(read(PROFILE))
    if fingerprint(PROFILE)!=release['profile_sha256']:raise ValueError('交付配置已改变；需重新验收，不能继续旧结果。')
    for name,digest in release['files'].items():
        if fingerprint(ROOT/name)!=digest:raise ValueError('交付文件已改变：'+name)
    return release


def best_validation(candidates,task):
    """No test score is read by this selector; incomplete candidates are excluded."""
    key='validation_accuracy' if task=='wn' else 'validation_mrr';available=[]
    for item in candidates:
        item=dict(item)
        if item.get('ledger_record'):
            ledger=read(ROOT/'artifacts/workflow_state/remaining_objectives_20261018_current.json')
            record=ledger['records'].get(item['ledger_record'],{})
            if record.get('state')!='completed':continue
            item['validation_run']=record['run_directory']
        directory=ROOT/item['validation_run']
        if not (directory/'summary.json').exists() or not (directory/'status.json').exists():continue
        summary=read(directory/'summary.json');state=read(directory/'status.json')['status']
        if not state.startswith('completed') or summary.get('seed',0)!=0:continue
        if summary.get('test_evaluations')!=0 or not math.isfinite(summary[key]):continue
        available.append((summary[key],item,summary))
    if not available:raise ValueError('尚无完整验证候选：'+task)
    _,item,summary=max(available,key=lambda row:row[0]);return item,summary


def freeze_selection(profile):
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    path=OUT/'protocol_selection.json'
    if path.exists():
        result=read(path)
        if result['release_sha256']!=fingerprint(RELEASE):raise ValueError('已冻结的正式重复协议属于另一交付版本。')
        for file,digest in result['input_sha256'].items():
            if fingerprint(file)!=digest:raise ValueError('已选验证输入改变：'+file)
        return path,result
    result=dict(release_sha256=fingerprint(RELEASE),selection_basis='maximum completed official validation; never test',
        historical_tests_seen=True,prior_0_5pp_diagnostic_gate_not_reclassified=True,formal_protocol_selection_is_new_user_authorized_stage=True,
        tasks={},input_sha256={})
    for task in ['wn','cloth']:
        item,s=best_validation(profile['candidates'][task],task);run=ROOT/item['validation_run'];config=read(run/'config.json')
        cache=Path(s.get('embedding_cache') or config['evaluation']['embedding_cache'])
        kind=item['training_kind'];recipe={}
        if task=='cloth':
            for key in ['head_variant','lr','batch_size','max_epochs','patience','step_lr_size','step_lr_gamma','loss_reduction','head_training_dropout']:
                if key in s:recipe[key]=s[key]
        result['tasks'][task]=dict(item,validation_run=str(run),dataset=s['dataset'],embedding_cache=str(cache),
            graph_policy=s['graph_policy'],recipe=recipe,selected_validation=s['validation_accuracy' if task=='wn' else 'validation_mrr'])
        directory=Path(read(run/'checkpoints/location.json')['checkpoint_directory'])
        for file in [run/'summary.json',run/'config.json',directory/('probe_best_raw_linear.pt' if task=='wn' else 'link_head_best.pt'),cache/'inference_state.json']:
            result['input_sha256'][str(file)]=fingerprint(file)
    books=profile['books'];run=ROOT/books['seed0_run'];s=read(run/'summary.json')
    recipe=dict(head_variant='existing_two_layer_head_upstream_schedule',lr=.001,batch_size=4096,max_epochs=100,patience=10,
        step_lr_size=None,step_lr_gamma=None,loss_reduction='separate_mean_sum')
    result['tasks']['books']=dict(dataset='books-lp',validation_run=str(run),embedding_cache=s['embedding_cache'],
        graph_policy='official_directed_train_only',training_kind='uniform_one',recipe=recipe,
        seed0_confirmed_run=str(run),paper_protocol_comparability_unresolved=True)
    result['tasks']['products']=dict(dataset='ogbn-products',validation_run=None,training_kind='linear',full_nodes=2449029,full_test_nodes=2213091)
    result['input_sha256'][str(run/'summary.json')]=fingerprint(run/'summary.json')
    atomic_write_json(path,result);return path,result


def audit_test(task,run):
    """Independent complete official IDs/labels/rank recomputation, without GPU."""
    import numpy as np
    import torch
    from came_repro.prepared_data import prepared_directory
    from came_repro.task_metrics import reverse_overlap_mask
    run=Path(run);summary=read(run/'summary.json')
    if summary.get('test_evaluations')!=1:raise ValueError('必须保存一次完整官方测试。')
    if task in ['cloth','books']:
        dataset='cloth-copurchase' if task=='cloth' else 'books-lp'
        split=torch.load(str(ROOT/'data/raw/mm_graph'/dataset/'lp-edge-split.pt'),map_location='cpu',weights_only=False,mmap=True)
        with np.load(run/'test_link_ranks.npz') as values:
            for key in ['source_node','target_node']:np.testing.assert_array_equal(values[key],split['test'][key])
            ranks=values['rank'];assert len(ranks)==len(split['test']['source_node']) and ((ranks>=1)&(ranks<=151)).all()
            score=float(np.mean(1./ranks.astype(np.float64)))
            mask=reverse_overlap_mask(np.asarray(split['train']['source_node']),np.asarray(split['train']['target_node']),
                np.asarray(split['test']['source_node']),np.asarray(split['test']['target_node']),125839 if task=='cloth' else 636502)
            strict=float(np.mean(1./ranks[~mask].astype(np.float64)))
        if abs(score-summary['test']['official']['mrr'])>1e-12:raise ValueError('保存的完整官方MRR与报告不一致。')
        return dict(score_pct=100*score,official_test_queries=len(ranks),reverse_overlap_queries=int(mask.sum()),reverse_nonoverlap_mrr_pct=100*strict)
    dataset='WN18RR' if task=='wn' else 'ogbn-products'
    with np.load(run/'test_predictions.npz') as values:
        if task=='wn':
            triples=np.load(prepared_directory(dataset)/'test_triples.npy',mmap_mode='r')
            np.testing.assert_array_equal(values['pairs'],triples[:,[0,2]]);np.testing.assert_array_equal(values['labels'],triples[:,1])
        else:
            ids=np.load(prepared_directory(dataset)/'test_indices.npy',mmap_mode='r')
            labels=np.load(prepared_directory(dataset)/'labels.npy',mmap_mode='r').reshape(-1)
            if len(ids)!=2213091:raise ValueError('Products必须完整官方测试。')
            np.testing.assert_array_equal(values['node_ids'],ids);np.testing.assert_array_equal(values['labels'],labels[ids])
        score=float(np.mean(values['prediction']==values['labels']));count=len(values['labels'])
    if score!=summary['test_accuracy']:raise ValueError('完整官方准确率与保存预测不一致。')
    return dict(score_pct=100*score,official_test_queries=count)


def wait_for_diagnostics(logger,pause):
    import psutil
    from came_repro.pause_control import TrainingPaused
    launch=ROOT/'artifacts/remaining_objectives_20261018/launch.json';last=None
    while launch.exists():
        if pause.exists():raise TrainingPaused('用户暂停交付入口。')
        value=read(launch)
        try:p=psutil.Process(value['pid']);alive=abs(p.create_time()-value['creation_time'])<1
        except psutil.NoSuchProcess:alive=False
        if not alive:return
        state=read(ROOT/'artifacts/workflow_state/remaining_objectives_20261018_current.json')['state']
        if state!=last:logger.info('WAIT_EXISTING_DIAGNOSTICS %s：保留当前任务，不重复启动GPU。',state);last=state
        time.sleep(15)


def run(preflight_only=False):
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.pause_control import TrainingPaused
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.storage_budget import check_budget
    from cloth_wn_autonomous import artifact_hashes
    verify_release();profile=read(PROFILE);OUT.mkdir(parents=True,exist_ok=True)
    guard=acquire_gpu_lock(ROOT/'.runtime/remaining_tasks_delivery.lock')
    ledger=read(LEDGER) if LEDGER.exists() else dict(release_sha256=fingerprint(RELEASE),state='ready',records={},active_seconds=0.,results={})
    if ledger['release_sha256']!=fingerprint(RELEASE):guard.close();raise ValueError('已运行交付版本不允许覆盖；请保留旧结果。')
    logger=logging.getLogger('remaining_delivery');logger.setLevel(logging.INFO)
    handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'pipeline.log',encoding='utf-8')]
    for h in handlers:
        formatter=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s');formatter.converter=time.gmtime;h.setFormatter(formatter);logger.addHandler(h)
    def persist():atomic_write_json(LEDGER,ledger)
    sleep_guard=False
    if sys.platform=='win32':
        import ctypes
        sleep_guard=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        if PAUSE.exists():
            old=read(PAUSE);history=OUT/'pause_history';history.mkdir(exist_ok=True)
            atomic_write_json(history/('resume_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),old)
            PAUSE.unlink()  # Only this entry's explicitly owned pause; no other queue flags touched.
            logger.info('RESUME：从已提交检查点继续；历史结果保留。')
        wait_for_diagnostics(logger,PAUSE)
        check_budget(additional_project_bytes=512*2**20,additional_cache_bytes=2**30)
        probe=acquire_gpu_lock();probe.close()
        def execute(jid,arguments,checkpoint=None):
            verify_release()
            if PAUSE.exists():raise TrainingPaused('用户暂停交付入口。')
            record=ledger['records'].setdefault(jid,dict(state='ready',attempts=[]))
            if record['state']=='completed':
                for path,digest in record['artifacts'].items():
                    if fingerprint(path)!=digest:raise ValueError('已完成结果改变：'+path)
                return record
            parent=best_resume(record,checkpoint) if checkpoint else None
            if jid.startswith('test_') and not parent:
                for attempt in record['attempts']:
                    prior=attempt.get('run_directory')
                    if prior and any((Path(prior)/name).exists() for name in ['test_predictions.npz','test_link_ranks.npz']):
                        raise ValueError('测试输出已存在但游标缺失；保留结果，需核查后继续。')
            if parent:arguments+=['--resume-head',str(parent)]
            ledger['state']='running_'+jid;persist();logger.info('JOB %s parent=%s',jid,parent)
            code,run=run_child(arguments,PAUSE,ledger,LEDGER,record,logger,float('inf'))
            if code==75 or PAUSE.exists():raise TrainingPaused('完整轮或编码游标已保存。')
            if code or not run:raise RuntimeError('子任务未完成：%s，退出码%s；原结果保留。'%(jid,code))
            record.update(state='completed',run_directory=run,summary=read(Path(run)/'summary.json'),artifacts=artifact_hashes(run));persist();return record
        worker=str(ROOT/'scripts/remaining_delivery_job.py')
        execute('acceptance',[worker,'--stage','acceptance'])
        execute('products_reference',[worker,'--stage','products_reference'])
        calibration=execute('products_calibration',[worker,'--stage','products_calibration'])
        ledger['products_capacity']=calibration['summary'];persist()
        if preflight_only:
            ledger['state']='preflight_passed' if calibration['summary']['ready'] else 'preflight_capacity_insufficient';persist();return ledger
        selection_file,selection=freeze_selection(profile)
        import numpy as np
        for task in ['cloth','wn','books','products']:
            if task=='products':
                if not calibration['summary']['ready']:
                    ledger['products_deferred']='精确数值或吞吐门槛未通过；未缩节点/维度/测试集。';persist();continue
                execute('products_extract',[worker,'--stage','products_extract'])
            item=selection['tasks'][task]
            for seed in range(10):
                key=task+'_s'+str(seed)
                if key in ledger['results']:continue
                if task=='books' and seed==0:
                    run=item['seed0_confirmed_run'];result=audit_test(task,run);result.update(reused=True,run_directory=run)
                else:
                    if seed==0 and task!='products':run=Path(item['validation_run'])
                    else:
                        record=execute('train_'+key,[worker,'--stage','train','--task',task,'--seed',str(seed),'--selection',str(selection_file)],
                            'probe_last.pt' if task in ['wn','products'] else 'link_head_last.pt')
                        run=Path(record['run_directory'])
                    summary=read(run/'summary.json');directory=Path(read(run/'checkpoints/location.json')['checkpoint_directory'])
                    head=directory/('probe_best_raw_linear.pt' if task=='wn' else 'probe_best.pt' if task=='products' else 'link_head_best.pt')
                    cache=Path(summary.get('embedding_cache') or item.get('embedding_cache'))
                    lock=dict(dataset=summary['dataset'],encoder_sha256=profile['encoder_sha256'],classifier_checkpoint=str(head),classifier_sha256=fingerprint(head),
                        validation_run_directory=str(run),validation_summary_sha256=fingerprint(run/'summary.json'),embedding_cache=str(cache),
                        inference_state_sha256=fingerprint(cache/'inference_state.json'),locked_before_test=True,test_used_for_selection=False)
                    if task=='wn':lock.update(graph_policy='train_reverse',feature_transform='train_pair_affine',selected_validation_accuracy=summary['validation_accuracy'])
                    elif task!='products':lock.update(head_variant=summary['head_variant'],selected_validation_mrr=summary['validation_mrr'])
                    lock_path=OUT/(key+'_selection_lock.json')
                    if lock_path.exists() and read(lock_path)!=lock:raise ValueError('已冻结测试选择锁改变。')
                    atomic_write_json(lock_path,lock)
                    args=[worker,'--stage','test','--task',task,'--selection',str(lock_path)]
                    tested=execute('test_'+key,args,'test_cursor.json');result=audit_test(task,tested['run_directory']);result.update(reused=False,validation_reused=seed==0 and task!='products',run_directory=tested['run_directory'])
                result.update(seed=seed,task=task,paper_pct=profile['paper_pct'][task],gap_pp=result['score_pct']-profile['paper_pct'][task])
                result['within_two_pp']=abs(result['gap_pp'])<=2
                if task=='books':result['paper_protocol_comparability_unresolved']=True
                ledger['results'][key]=result;persist();write_results(ledger,profile)
                logger.info('RESULT %s seed=%d score=%.6f%% paper=%.2f%% gap=%+.6fpp; all seeds retained',task,seed,result['score_pct'],result['paper_pct'],result['gap_pp'])
        ledger['state']='completed_main_repeats' if len(ledger['results'])==40 else 'partial_products_capacity_blocked';persist();write_results(ledger,profile)
        return ledger
    except BaseException as error:
        ledger.update(state='paused' if isinstance(error,(TrainingPaused,KeyboardInterrupt)) or PAUSE.exists() else 'failed',error=str(error));persist();raise
    finally:
        if sleep_guard:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for h in handlers:h.close();logger.removeHandler(h)
        guard.close()


def write_results(ledger,profile):
    import numpy as np
    from came_repro.atomic_io import atomic_write_json
    rows=list(ledger['results'].values());statistics={}
    for task in ['cloth','wn','books','products']:
        values=np.array([r['score_pct'] for r in rows if r['task']==task])
        if len(values):statistics[task]=dict(completed=len(values),expected=10,mean_pct=float(values.mean()),sample_sd_pp=float(values.std(ddof=1)) if len(values)>1 else None,
            gap_pp=float(values.mean()-profile['paper_pct'][task]),all_seeds_retained=True,paper_protocol_comparability_unresolved=task=='books')
    atomic_write_json(OUT/'results.json',dict(state=ledger['state'],statistics=statistics,results=rows,shared_pretraining_checkpoints=1,paper_reproduction_complete=False))
    with (OUT/'results.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=['task','seed','score_pct','paper_pct','gap_pp','within_two_pp','reused','run_directory'],extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,tasks,ylabel in [(axes[0],['wn','products'],'Accuracy (%)'),(axes[1],['cloth','books'],'MRR x 100')]:
            available=[task for task in tasks if task in statistics]
            if not available:ax.text(.5,.5,'Pending',ha='center',va='center',transform=ax.transAxes);continue
            x=np.arange(len(available));means=[statistics[t]['mean_pct'] for t in available]
            sd=[statistics[t]['sample_sd_pp'] or 0 for t in available]
            ax.bar(x-.18,means,.36,yerr=sd,capsize=3,color='#315a7d',label='Observed mean / sample SD')
            ax.bar(x+.18,[profile['paper_pct'][t] for t in available],.36,color='#c4c4c4',label='Paper')
            ax.set_xticks(x);ax.set_xticklabels([t+' (n='+str(statistics[t]['completed'])+'/10)' for t in available]);ax.set_ylabel(ylabel)
            ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=7)
        fig.text(.02,.01,'Full official test; all completed seeds retained; one shared pretraining checkpoint. Books paper protocol remains unresolved.',fontsize=8)
        fig.tight_layout(rect=(0,.08,1,1));directory=OUT/'figures';directory.mkdir(exist_ok=True)
        fig.savefig(directory/'remaining_main_comparison.png',dpi=180);plt.close(fig)
