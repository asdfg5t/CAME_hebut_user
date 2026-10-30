"""WN18RR主任务06：准备检查与一次seed0确认后，用户点击完成十seed。"""
import argparse
import copy
import csv
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
OUT=ROOT/'artifacts/wn_final_repeats'
PROFILE=ROOT/'configs/wn_final_repeats_20261021.json'
RELEASE=ROOT/'artifacts/delivery/wn_final_repeats_release.json'
LEDGER=ROOT/'artifacts/workflow_state/wn_final_repeats_current.json'
PAUSE=ROOT/'artifacts/workflow_state/wn_final_repeats_pause.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_profile(p):
    expected=dict(schema='came_wn_final_repeats_v1',dataset='WN18RR',task='relation_classification',nodes=40943,width=1024,classes=11,
        official_train_rows=86835,official_validation_rows=3034,official_test_rows=3134,metric_admission_gate=False,
        max_new_seed0_tests=1,training_kind='normalized_adam',optimizer='Adam',lr=.05,max_epochs=2000,
        patience=100,training_batch_size=0,graph_policy='train_reverse',edge_representation='head_tail_concatenation')
    if any(p.get(k)!=v for k,v in expected.items()) or p.get('seeds')!=list(range(10)):
        raise ValueError('必须固定WN官方完整划分、1024、线性关系分类、全部seed及原验证选定方案。')


def verify():
    from came_repro.author_loader import fingerprint,verify_originals
    from remaining_delivery import verify_release
    verify_release();verify_originals();p=read(PROFILE);validate_profile(p)
    for path,digest in read(RELEASE)['files'].items():
        file=Path(path);file=file if file.is_absolute() else ROOT/file
        if fingerprint(file)!=digest:raise ValueError('WN冻结输入改变：'+path)
    run=ROOT/p['validation_run'];s=read(run/'summary.json')
    if (fingerprint(run/'summary.json')!=p['validation_summary_sha256'] or s['seed']!=0
            or s['test_evaluations']!=0 or not read(run/'status.json')['status'].startswith('completed')):
        raise ValueError('Seed0必须复用完整且仅验证选定的保存头。')
    return p


def selection_file(p):
    from came_repro.atomic_io import atomic_write_json
    run=ROOT/p['validation_run'];cache=read(run/'config.json')['evaluation']['embedding_cache']
    value=dict(tasks=dict(wn=dict(validation_run=str(run),dataset='WN18RR',embedding_cache=cache,
        training_kind='normalized_adam',graph_policy='train_reverse')),selection_basis=p['selection_basis'],test_used_for_selection=False)
    path=OUT/'protocol_selection.json'
    if path.exists() and read(path)!=value:raise ValueError('正式重复选择锁不可改变。')
    atomic_write_json(path,value);return path


def lock_test(run):
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    run=Path(run);s=read(run/'summary.json');cache=Path(read(run/'config.json')['evaluation']['embedding_cache'])
    head=Path(read(run/'checkpoints/location.json')['checkpoint_directory'])/'probe_best_raw_linear.pt'
    value=dict(dataset='WN18RR',encoder_sha256=s['encoder_sha256'],classifier_checkpoint=str(head),classifier_sha256=fingerprint(head),
        validation_run_directory=str(run),validation_summary_sha256=fingerprint(run/'summary.json'),embedding_cache=str(cache),
        inference_state_sha256=fingerprint(cache/'inference_state.json'),locked_before_test=True,test_used_for_selection=False,
        graph_policy='train_reverse',feature_transform='train_pair_affine',selected_validation_accuracy=s['validation_accuracy'])
    path=OUT/('seed'+str(s['seed'])+'_test_lock.json')
    if path.exists() and read(path)!=value:raise ValueError('已锁测试头不能替换。')
    atomic_write_json(path,value);return path


def preflight():
    import torch
    import numpy as np
    from came_repro.protocol_repair import check_cuda
    from came_repro.experiment import Experiment,seed_everything
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro.prepared_data import prepared_directory
    from came_repro.kg_graph_probe import kg_input
    from came_repro.kg_selected_confirmation import RawDoubleInput
    from came_repro import downstream
    from cloth_wn_autonomous_backend import assert_same
    p=verify();check_cuda();torch.set_num_threads(4);seed_everything(713)
    storage=read(ROOT/'configs/storage.json');x=torch.randn(32,8,device='cuda');y=(x[:,0]>0).long()
    class Fixture:
        width=8
        labels=y.cpu().numpy()
        indices=np.arange(32)
        pairs=None
        def __len__(self):return 32
        def batch(self,index):return x[index].clone(),y[index].clone()
    fixture=Fixture();config=dict(name='wn_final_resume_fixture',seed=1,storage=storage,evaluation=dict(fixture_only_not_benchmark=True,dataset='synthetic32row8feature_fixture'))
    runs=[];original=downstream.pause_requested
    for mode in ['continuous','paused','resumed']:
        parent=runs[1].checkpoint_directory/'probe_last.pt' if mode=='resumed' else None
        e=Experiment(copy.deepcopy(config),dict(fixture_only_not_benchmark=True),parent);runs.append(e);calls=[0]
        def pause_two():calls[0]+=1;return calls[0]>=2
        if mode=='paused':downstream.pause_requested=pause_two
        try:
            try:downstream.train_linear_probe(fixture,fixture,None,2,3,100,1,e,lr=.05,resume=parent,evaluate_test=False)
            except downstream.TrainingPaused:
                if mode!='paused':raise
                e.status('paused_fixture')
            else:
                if mode=='paused':raise ValueError('Pause fixture did not pause')
                e.status('completed_fixture')
        finally:downstream.pause_requested=original;e.close()
    a,b=[torch.load(str(runs[i].checkpoint_directory/'probe_last.pt'),map_location='cpu',weights_only=False) for i in [0,2]]
    for key in ['classifier','optimizer','rng','sampling_generator','history_rows','best_state','selected_epoch','next_epoch']:assert_same(a[key],b[key])
    if a['next_epoch']!=4:raise ValueError('Incomplete resume fixture')
    del a,b,x,y;torch.cuda.empty_cache()
    run=ROOT/p['validation_run'];s=read(run/'summary.json');cache=Path(read(run/'config.json')['evaluation']['embedding_cache']);state=read(cache/'inference_state.json')
    if not state['complete'] or state['identity']['checkpoint_sha256']!=p['encoder_sha256'] or fingerprint(cache/'fused_embeddings.npy')!=state['output_sha256']:
        raise ValueError('Complete frozen normalized cache required')
    matrix=np.load(cache/'fused_embeddings.npy',mmap_mode='r')
    if matrix.shape!=(40943,1024):raise ValueError('Full40943x1024 cache required')
    gpu=torch.from_numpy(np.array(matrix,copy=True)).cuda()
    if not torch.isfinite(gpu).all():raise ValueError('Nonfinite WN cache')
    norm=gpu.norm(dim=1);nonzero=norm>1e-8
    if torch.max(torch.abs(norm[nonzero]-1.)).item()>1e-5:raise ValueError('Selected unitL2 representation changed')
    del gpu,norm;torch.cuda.empty_cache()
    directory=prepared_directory('WN18RR');manifest=read(directory/'manifest.json')
    if fingerprint(directory/'manifest.json')!=state['identity']['data']['prepared_manifest_sha256']:raise ValueError('Official manifest changed')
    for file,digest in manifest['files'].items():
        if fingerprint(directory/file)!=digest:raise ValueError('Official KG asset changed')
    for split,count in [('train',86835),('valid',3034),('test',3134)]:
        if len(np.load(directory/(split+'_triples.npy'),mmap_mode='r'))!=count:raise ValueError('Full official KG split required')
    raw=kg_input('WN18RR',matrix,'valid');valid=RawDoubleInput(matrix,raw.labels,pairs=raw.pairs)
    head_path=Path(read(run/'checkpoints/location.json')['checkpoint_directory'])/'probe_best_raw_linear.pt'
    head=torch.nn.Linear(2048,11,dtype=torch.float64).cuda();head.load_state_dict(torch.load(str(head_path),map_location='cpu',weights_only=False)['classifier'])
    accuracy,_,prediction=downstream.score_probe(head,valid,save_predictions=True)
    with np.load(run/'validation_predictions.npz') as values:np.testing.assert_array_equal(prediction,values['prediction'])
    if accuracy!=s['validation_accuracy']:raise ValueError('Full validation replay changed')
    e=Experiment(dict(name='wn_final_runtime_check',seed=0,storage=storage,evaluation=dict(fixture_only=False,test_evaluations=0,
        validation_rows=3034,release_sha256=fingerprint(RELEASE))),state['identity']['data'],head_path)
    try:
        np.savez_compressed(e.directory/'validation_predictions.npz',prediction=prediction,labels=raw.labels,pairs=raw.pairs)
        result=dict(passed=True,actual_Torch_DGL_CUDA=True,full_Adam_RNG_sampler_history_resume_verified=True,
            fixture_runs=[str(r.directory) for r in runs],full_official_validation_rows=3034,validation_accuracy=accuracy,
            all_official_splits_verified=True,unitL2_full_cache_verified=True,test_evaluations=0,release_sha256=fingerprint(RELEASE))
        e.metric(dict(step=1,validation_accuracy=accuracy));atomic_write_json(e.directory/'summary.json',result);e.status('completed_wn_runtime_check')
    finally:e.close()


def summarize(ledger,p):
    import numpy as np
    from came_repro.atomic_io import atomic_write_json
    rows=sorted(ledger['results'].values(),key=lambda r:r['seed']);scores=np.array([r['score_pct'] for r in rows])
    stats=dict(completed=len(rows),expected=10,paper_pct=p['paper_accuracy_pct'],all_seeds_retained=True,
        metric_admission_gate=False,paper_reproduction_complete=False,shared_pretraining_checkpoints=1)
    if len(rows):stats.update(mean_pct=float(scores.mean()),sample_sd_pp=float(scores.std(ddof=1)) if len(rows)>1 else None,gap_pp=float(scores.mean()-p['paper_accuracy_pct']))
    atomic_write_json(OUT/'results.json',dict(state=ledger['state'],statistics=stats,results=rows))
    with (OUT/'results.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=['seed','score_pct','paper_pct','gap_pp','official_test_queries','run_directory']);writer.writeheader();writer.writerows(rows)
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4));ids=[r['seed'] for r in rows]
        ax.plot(ids,scores,'o-',color='#315a7d',label='All completed seeds');ax.axhline(p['paper_accuracy_pct'],color='black',ls='--',label='Paper81.97')
        ax.set(xticks=ids,xlabel='Downstream initialization seed',ylabel='Official test relation accuracy (%)',title='WN18RR: %d/10 results'%len(rows));ax.legend(fontsize=8)
        fig.text(.02,.01,'Source: results.csv; full3134 test triples; one frozen encoder; nodeL2/train-only affine engineering adaptation; all seeds retained.',fontsize=7)
        fig.tight_layout(rect=(0,.08,1,1));fig.savefig(OUT/'wn_final_comparison.png',dpi=180);plt.close(fig)


def pipeline(prepare_only=False,check_only=False):
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.pause_control import TrainingPaused
    from came_repro.storage_budget import check_budget
    from cloth_wn_autonomous import artifact_hashes
    from remaining_delivery import audit_test
    p=verify();OUT.mkdir(parents=True,exist_ok=True);guard=acquire_gpu_lock(ROOT/'.runtime/wn_final_repeats.lock')
    ledger=read(LEDGER) if LEDGER.exists() else dict(release_sha256=fingerprint(RELEASE),state='ready',records={},results={},active_seconds=0.)
    logger=logging.getLogger('wn_final_repeats');logger.setLevel(logging.INFO)
    handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'pipeline.log',encoding='utf-8')]
    for h in handlers:
        fmt=logging.Formatter('%(asctime)s UTC %(levelname)s %(message)s');fmt.converter=time.gmtime;h.setFormatter(fmt);logger.addHandler(h)
    def persist():atomic_write_json(LEDGER,ledger)
    sleep_guard=False
    try:
        if ledger['release_sha256']!=fingerprint(RELEASE):raise ValueError('Cannot replace a frozen existing workflow')
        if PAUSE.exists():
            import datetime
            history=OUT/'pause_history';history.mkdir(exist_ok=True)
            atomic_write_json(history/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),read(PAUSE));PAUSE.unlink()
        if os.name=='nt':
            import ctypes
            sleep_guard=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        probe=acquire_gpu_lock();probe.close();check_budget(**p['budget']);selection=selection_file(p)
        def execute(name,args,filename=None):
            verify()
            if PAUSE.exists():raise TrainingPaused('User paused WN')
            rec=ledger['records'].setdefault(name,dict(state='ready',attempts=[]))
            if rec['state']=='completed':
                for file,digest in rec['artifacts'].items():
                    if fingerprint(file)!=digest:raise ValueError('Completed result changed')
                return rec
            parent=best_resume(rec,filename) if filename else None
            if name.startswith('test_') and not parent:
                for attempt in rec['attempts']:
                    old=attempt.get('run_directory')
                    if old and (Path(old)/'test_predictions.npz').exists():raise ValueError('Test output without consistent cursor requires audit')
            if parent:args+=['--resume-head',str(parent)]
            ledger['state']='running_'+name;persist();logger.info('JOB %s parent=%s',name,parent)
            code,run=run_child(args,PAUSE,ledger,LEDGER,rec,logger,float('inf'))
            if code==75 or PAUSE.exists():raise TrainingPaused('Epoch/test cursor saved')
            if code or not run:raise RuntimeError('WN child did not complete: '+name+' exit='+str(code))
            rec.update(state='completed',run_directory=run,summary=read(Path(run)/'summary.json'),artifacts=artifact_hashes(run));persist();return rec
        accepted=execute('runtime_check',[str(Path(__file__)),'--stage','check'])
        if not accepted['summary']['passed']:raise ValueError('Actual CUDA/resume/full-validation check required')
        if check_only:ledger['state']='runtime_check_passed';persist();return ledger
        worker=str(ROOT/'scripts/remaining_delivery_job.py')
        for seed in range(10):
            key=str(seed)
            if key in ledger['results']:
                for file,digest in ledger['records']['test_s'+key]['artifacts'].items():
                    if fingerprint(file)!=digest:raise ValueError('Saved official result changed')
            else:
                run=ROOT/p['validation_run'] if seed==0 else Path(execute('train_s'+key,[worker,'--stage','train','--task','wn','--seed',key,'--selection',str(selection)],'probe_last.pt')['run_directory'])
                lock=lock_test(run);tested=execute('test_s'+key,[worker,'--stage','test','--task','wn','--selection',str(lock)],'test_cursor.json')
                result=audit_test('wn',tested['run_directory'])
                if result['official_test_queries']!=3134:raise ValueError('Complete official WN test required')
                result.update(seed=seed,run_directory=tested['run_directory'],paper_pct=p['paper_accuracy_pct'],gap_pp=result['score_pct']-p['paper_accuracy_pct'])
                ledger['results'][key]=result;persist();summarize(ledger,p)
                logger.info('WN_RESULT %d/10 seed=%d test_accuracy=%.8f%% gap=%+.8fpp; retained',len(ledger['results']),seed,result['score_pct'],result['gap_pp'])
            if prepare_only and seed==0:
                ledger['state']='ready_for_user_repeats';persist();summarize(ledger,p);return ledger
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dry-run',action='store_true');parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--check-only',action='store_true');parser.add_argument('--pause',action='store_true');parser.add_argument('--stage',choices=['check'])
    args=parser.parse_args()
    for stream in [sys.stdout,sys.stderr]:
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    if args.pause:atomic_write_json(PAUSE,dict(reason='user_requested_WN_pause'));print('已请求完整轮/测试游标边界保存；下次点同文件继续。');return
    verify()
    if args.stage:preflight();return
    print('WN18RR主任务06：固定已完成验证选定方案，十seed关系分类，实际CUDA；支持断点续训。',flush=True)
    print('先检测/复用已完成seed0；默认接续seed1..9；不按测试分数停训，不重编码图。',flush=True)
    print('日志/结果：'+str(OUT),flush=True)
    if args.dry_run:print('冻结身份核验通过；GPU任务0。');return
    for name in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']:os.environ[name]='1'
    result=pipeline(args.prepare_only,args.check_only);print('WN状态：'+result['state'],flush=True)


if __name__=='__main__':main()
