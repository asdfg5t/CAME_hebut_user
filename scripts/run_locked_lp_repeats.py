"""Repeat only LP protocols that passed independently verified full-test delivery gates."""
import argparse
import copy
import sys
from pathlib import Path

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
RECIPE_KEYS=('head_variant','lr','batch_size','max_epochs','patience','step_lr_size','step_lr_gamma','loss_reduction')


def train_repeat(gpu,train,valid,recipe,seed,experiment,resume=None):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from came_repro import link_evaluation as link
    from came_repro.lp_validation_repair import make_head,TrainOnlyNegatives
    from came_repro.experiment import seed_everything,rng_state,restore_rng,atomic_checkpoint
    from came_repro.task_metrics import reverse_overlap_mask
    from came_repro.pause_control import pause_requested,TrainingPaused
    seed_everything(seed);random=np.random.default_rng(seed)
    source,target=(np.asarray(train[k],dtype=np.int64) for k in ('source_node','target_node'))
    sampler=TrainOnlyNegatives(source,target,len(gpu));overlap=reverse_overlap_mask(source,target,np.asarray(valid['source_node']),np.asarray(valid['target_node']),len(gpu))
    head=make_head(recipe['head_variant']).cuda();opt=torch.optim.Adam(head.parameters(),lr=recipe['lr'])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,recipe['step_lr_size'],recipe['step_lr_gamma']) if recipe['step_lr_size'] else None
    best,selected,stale,first,history,best_state=-1.,0,0,1,[],None
    if resume:
        saved=torch.load(str(resume),map_location='cpu',weights_only=False)
        if saved['schema']!='came_locked_lp_repeat_v1' or saved['config']!=experiment.config or saved['data_audit']!=experiment.data_audit or saved['seed']!=seed:raise ValueError('Locked LP repeat identity changed')
        head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer'])
        if scheduler:scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng']);random.bit_generator.state=saved['sampling_generator']
        best,selected,stale,first,history,best_state=saved['best_mrr'],saved['selected_epoch'],saved['stale'],saved['next_epoch'],saved['history_rows'],saved['best_state']
        for row in history:experiment.metric(row)
    for epoch in range(first,first if stale>=recipe['patience'] else recipe['max_epochs']+1):
        head.train();order=random.permutation(len(source));total=0.;used_lr=opt.param_groups[0]['lr']
        for start in range(0,len(order),recipe['batch_size']):
            ids=order[start:start+recipe['batch_size']];negative=sampler.sample(source[ids],random)
            a,b,c=(link.vectors(gpu,x) for x in (source[ids],target[ids],negative))
            opt.zero_grad(set_to_none=True);positive,negative_score=head(a,b),head(a,c)
            if recipe['loss_reduction']=='separate_mean_sum':loss=F.binary_cross_entropy_with_logits(positive,torch.ones_like(positive))+F.binary_cross_entropy_with_logits(negative_score,torch.zeros_like(negative_score))
            else:loss=F.binary_cross_entropy_with_logits(torch.cat((positive,negative_score)),torch.cat((torch.ones_like(positive),torch.zeros_like(negative_score))))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite locked LP loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('Nonfinite locked LP gradient')
            opt.step();total+=loss.item()*len(ids)
        if scheduler:scheduler.step()
        metric=link.evaluate_split(head,gpu,valid,64,overlap)['official']['mrr']
        row={'step':epoch,'train_loss':total/len(source),'validation_mrr':metric,'lr_used':used_lr,'lr_next':opt.param_groups[0]['lr']};history.append(row);experiment.metric(row)
        if metric>best:
            best,selected,stale=metric,epoch,0;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else:stale+=1
        experiment.logger.info('Locked LP seed=%d epoch=%d train=%.6f validation_MRR=%.8f selected=%d',seed,epoch,row['train_loss'],metric,selected)
        atomic_checkpoint(experiment.checkpoint_directory/'link_head_last.pt',{'schema':'came_locked_lp_repeat_v1','seed':seed,'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,'rng':rng_state(),'sampling_generator':random.bit_generator.state,'completed_epochs':epoch,'next_epoch':epoch+1,'best_mrr':best,'selected_epoch':selected,'stale':stale,'best_state':best_state,'history_rows':history,'config':experiment.config,'data_audit':experiment.data_audit,'cursor_policy':'full official train-positive epoch boundary'})
        if pause_requested():raise TrainingPaused('Locked LP full checkpoint saved')
        if stale>=recipe['patience']:break
    head.load_state_dict(best_state)
    replay=link.evaluate_split(head,gpu,valid,64,overlap,experiment.directory/'validation_link_ranks.npz')
    if abs(replay['official']['mrr']-best)>1e-12:raise ValueError('Selected LP validation replay mismatch')
    atomic_checkpoint(experiment.checkpoint_directory/'link_head_best.pt',{'head':best_state,'selected_epoch':selected,'validation_mrr':best})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));steps=[r['step'] for r in history]
    axes[0].plot(steps,[r['train_loss'] for r in history]);axes[0].set(xlabel='Full train-positive epoch',ylabel='Binary cross entropy')
    axes[1].plot(steps,[r['validation_mrr'] for r in history]);axes[1].axvline(selected,ls='--',color='black');axes[1].set(xlabel='Full train-positive epoch',ylabel='Official validation MRR')
    fig.text(.02,.01,'Fixed LP protocol; seed=%d; train-only graph; full validation selects saved head; test held out.'%seed,fontsize=8);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(experiment.directory/'figures/locked_lp_curves.png',dpi=180);plt.close(fig)
    return dict(recipe,seed=seed,selected_epoch=selected,epochs_executed=len(history),validation_mrr=best,test=None,test_evaluations=0)


def main():
    import json,math,datetime,logging
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file,atomic_write_json
    from came_repro.author_loader import fingerprint,verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.experiment import Experiment,write_json
    from came_repro.protocol_repair import check_cuda
    from came_repro.lp_validation_repair import load_verified_cache
    from came_repro.remaining_lp_pilot import use_cached_vectors
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--train-one',action='store_true');parser.add_argument('--acceptance',action='store_true');parser.add_argument('--dry-run',action='store_true');parser.add_argument('--dataset',choices=['sports-copurchase','cloth-copurchase']);parser.add_argument('--seed',type=int);parser.add_argument('--reference-run',type=Path);parser.add_argument('--resume-head',type=Path);args=parser.parse_args()
    if args.acceptance:
        from came_repro.protocol_repair import train_lp
        from came_repro.experiment import seed_everything
        check_cuda();torch.set_num_threads(4);seed_everything(216)
        gpu=torch.randn(12,1024,device='cuda');train={'source_node':np.array([0,1,2,3,4,5]),'target_node':np.array([1,2,3,4,5,6])};valid={'source_node':np.array([6,7]),'target_node':np.array([7,8]),'target_node_neg':np.tile(np.arange(12),(2,13))[:,:150]}
        recipe={'head_variant':'existing_two_layer_head_upstream_schedule','lr':.001,'batch_size':3,'max_epochs':2,'patience':10,'step_lr_size':None,'step_lr_gamma':None,'loss_reduction':'separate_mean_sum'}
        config={'name':'locked_lp_acceptance','seed':0,'storage':read_json_file(ROOT/'configs/storage.json'),'evaluation':{'dataset':'fixture_not_paper'}}
        baseline=Experiment(config,{'fixture_only':True});candidate=Experiment(config,{'fixture_only':True})
        try:
            with use_cached_vectors():a=train_lp(gpu,train,valid,recipe,baseline);b=train_repeat(gpu,train,valid,recipe,0,candidate)
            x=torch.load(str(baseline.checkpoint_directory/'link_head_last.pt'),map_location='cpu',weights_only=False);y=torch.load(str(candidate.checkpoint_directory/'link_head_last.pt'),map_location='cpu',weights_only=False)
            assert x['history_rows']==y['history_rows'] and a['validation_mrr']==b['validation_mrr']
            for name in x['head']:assert torch.equal(x['head'][name],y['head'][name])
            baseline.status('completed_acceptance_fixture');candidate.status('completed_acceptance_fixture')
            import came_repro.pause_control as pauses
            seed_recipe=dict(recipe,max_epochs=3)
            seed_config=dict(config,name='locked_lp_seed1_resume_acceptance',seed=1)
            full=Experiment(seed_config,{'fixture_only':True})
            try:
                with use_cached_vectors():train_repeat(gpu,train,valid,seed_recipe,1,full)
                full.status('completed_acceptance_fixture')
            finally:full.close()
            paused=Experiment(seed_config,{'fixture_only':True})
            original_pause=pauses.pause_requested;calls=[0]
            def pause_at_two():calls[0]+=1;return calls[0]==2
            pauses.pause_requested=pause_at_two
            try:
                try:
                    with use_cached_vectors():train_repeat(gpu,train,valid,seed_recipe,1,paused)
                except pauses.TrainingPaused:paused.status('paused_acceptance_fixture')
                else:raise ValueError('Seed1 checkpoint pause fixture failed')
            finally:pauses.pause_requested=original_pause;paused.close()
            saved=paused.checkpoint_directory/'link_head_last.pt'
            resumed=Experiment(seed_config,{'fixture_only':True},saved)
            try:
                with use_cached_vectors():train_repeat(gpu,train,valid,seed_recipe,1,resumed,saved)
                resumed.status('completed_acceptance_fixture')
            finally:resumed.close()
            first=torch.load(str(full.checkpoint_directory/'link_head_last.pt'),map_location='cpu',weights_only=False)
            last=torch.load(str(resumed.checkpoint_directory/'link_head_last.pt'),map_location='cpu',weights_only=False)
            assert first['history_rows']==last['history_rows'] and first['next_epoch']==last['next_epoch']==4
            for name in first['head']:assert torch.equal(first['head'][name],last['head'][name])
            report={'passed':True,'actual_Torch_DGL_CUDA':True,'seed0_equivalent_to_validated_candidate_bitexact':True,'seed1_pause_resume_bitexact_weights_history':True,'resume_parent_sha256':fingerprint(saved),'script_sha256':fingerprint(__file__),'fixture_runs':[str(e.directory) for e in [baseline,candidate,full,paused,resumed]]}
            write_json(ROOT/'artifacts/audit/locked_lp_repeat_CUDA_acceptance_20261014.json',report);print(json.dumps(report))
        finally:baseline.close();candidate.close()
        return
    if args.train_one:
        check_cuda();torch.set_num_threads(4)
        if args.seed not in range(1,10):raise ValueError('Only the predeclared new seeds1-9 may train')
        accepted=read_json_file(ROOT/'artifacts/audit/locked_lp_repeat_CUDA_acceptance_20261014.json')
        if not accepted['passed'] or accepted['script_sha256']!=fingerprint(__file__):raise ValueError('Accepted frozen repeat implementation required')
        reference=read_json_file(args.reference_run/'summary.json');cache=Path(reference['embedding_cache']);digest=reference['encoder_sha256']
        gpu,audit=load_verified_cache(args.dataset,cache,digest);recipe={k:reference[k] for k in RECIPE_KEYS}
        config=read_json_file(args.reference_run/'config.json');config.update(name='locked_lp_'+args.dataset,seed=args.seed);config['evaluation'].update(seed=args.seed,locked_repeat_entry_sha256=fingerprint(__file__),test_used_for_selection=False)
        encoder=Path(read_json_file(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
        experiment=Experiment(config,audit,args.resume_head or encoder)
        try:
            split=torch.load(str(ROOT/'data/raw/mm_graph'/args.dataset/'lp-edge-split.pt'),map_location='cpu',weights_only=False,mmap=True)
            with use_cached_vectors():summary=train_repeat(gpu,split['train'],split['valid'],recipe,args.seed,experiment,args.resume_head)
            summary.update(dataset=args.dataset,task='link',representation='came',encoder_sha256=digest,embedding_cache=str(cache),graph_policy='upstream_train_bidirected_unique',diagnostic_only=False,paper_reproduction_complete=False)
            write_json(experiment.directory/'summary.json',summary);experiment.status('completed_locked_lp_seed_validation')
        except BaseException as error:
            from came_repro.pause_control import TrainingPaused
            experiment.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
        finally:experiment.close()
        return
    pointer=ROOT/'artifacts/remaining_main_pilots/two_lp_priority/verified_delivery.json'
    if not pointer.exists():print('NOT_READY: 两项仍在单seed验收；没有达标交付声明，不启动正式训练。',flush=True);return
    delivery=read_json_file(pointer)
    if not delivery['tasks']:print('NOT_READY: 当前没有通过完整测试目标与独立验收的LP任务。',flush=True);return
    accepted=read_json_file(ROOT/'artifacts/audit/locked_lp_repeat_CUDA_acceptance_20261014.json')
    if not accepted['passed'] or accepted['script_sha256']!=fingerprint(__file__):raise ValueError('Repeat CUDA acceptance changed')
    print('VERIFIED LP TASKS: '+', '.join(delivery['tasks'])+'; reuse seed0; each task trains seeds1-9 only.',flush=True)
    if args.dry_run:print('DRY_RUN: GPU jobs=0.',flush=True);return
    from came_repro.quality_pipeline import run_child,best_resume
    from came_repro.all_unmet_repair_pipeline import collect_protocol_artifacts
    from came_repro.storage_budget import check_budget
    try:gpu=acquire_gpu_lock();gpu.close()
    except RuntimeError:print('GPU_BUSY: 当前单seed诊断仍在运行，完成后再点击；未启动正式重复。',flush=True);return
    guard=acquire_gpu_lock(ROOT/'.runtime/locked_lp_repeats.lock')
    directory=ROOT/'artifacts/locked_lp_repeats';directory.mkdir(parents=True,exist_ok=True)
    path=ROOT/'artifacts/workflow_state/locked_lp_repeats_current.json';pause=ROOT/'artifacts/workflow_state/locked_lp_repeats_pause.json'
    identity={'delivery_sha256':fingerprint(pointer),'script_sha256':fingerprint(__file__),'sources':{str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}}
    ledger=read_json_file(path) if path.exists() else {'identity':identity,'state':'ready','records':{},'active_seconds':0.,'statistics':{}}
    if ledger['identity']!=identity:guard.close();raise ValueError('Locked LP repeat identity changed')
    logger=logging.getLogger('locked_lp_repeats');logger.setLevel(logging.INFO)
    for handler in [logging.StreamHandler(),logging.FileHandler(directory/('pipeline_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log'),encoding='utf-8')]:handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logger.addHandler(handler)
    try:
        verify_originals();check_budget(additional_cache_bytes=256*2**20)
        def execute(jid,arguments,resume_name=None):
            if fingerprint(pointer)!=identity['delivery_sha256'] or fingerprint(__file__)!=identity['script_sha256']:raise ValueError('Locked LP delivery changed')
            for name,digest in identity['sources'].items():
                if fingerprint(ROOT/name)!=digest:raise ValueError('Locked LP production source changed')
            rec=ledger['records'].setdefault(jid,{'state':'ready','attempts':[]})
            if rec['state']=='completed':
                for name,digest in rec['artifact_sha256'].items():
                    if fingerprint(name)!=digest:raise ValueError('Completed repeat artifact changed')
                return rec
            if resume_name:
                saved=best_resume(rec,resume_name)
                if saved:arguments+=['--resume-head',str(saved)]
            ledger['state']='running';code,run=run_child(arguments,pause,ledger,path,rec,logger,float('inf'))
            if code or not run:
                ledger['state']=rec['state']='paused' if code==75 else 'failed';atomic_write_json(path,ledger);raise RuntimeError('Locked LP child stopped')
            rec.update(state='completed',run_directory=run,summary=read_json_file(Path(run)/'summary.json'),artifact_sha256=collect_protocol_artifacts(run));atomic_write_json(path,ledger);return rec
        for dataset in delivery['tasks']:
            item=delivery['tasks'][dataset];seed0=read_json_file(Path(item['confirmation_run'])/'summary.json');reference=Path(seed0['validation_run_directory']);scores=[seed0['test']['official']['mrr']*100]
            if abs(scores[0]-item['paper_mrr_pct'])>2 or fingerprint(Path(item['confirmation_run'])/'summary.json')!=item['confirmation_summary_sha256']:raise ValueError('Delivered seed0 changed or does not meet target')
            for seed in range(1,10):
                logger.info('MAIN_TASK %s DATASET=%s NEW_SEED %d/9',item['main_task_number'],dataset,seed)
                rec=execute(dataset+'_validation_seed'+str(seed),['scripts/run_locked_lp_repeats.py','--train-one','--dataset',dataset,'--seed',str(seed),'--reference-run',str(reference)],'link_head_last.pt')
                run=Path(rec['run_directory']);s=rec['summary'];cache=Path(s['embedding_cache']);best=Path(read_json_file(run/'checkpoints/location.json')['checkpoint_directory'])/'link_head_best.pt'
                selected={'dataset':dataset,'encoder_sha256':s['encoder_sha256'],'classifier_checkpoint':str(best),'classifier_sha256':fingerprint(best),'validation_run_directory':str(run),'validation_summary_sha256':fingerprint(run/'summary.json'),'embedding_cache':str(cache),'inference_state_sha256':fingerprint(cache/'inference_state.json'),'locked_before_test':True,'test_used_for_selection':False,'selected_validation_mrr':s['validation_mrr'],'head_variant':s['head_variant'],'graph_policy':'upstream_train_bidirected_unique'}
                lock_path=directory/(dataset+'_seed'+str(seed)+'_selection_lock.json')
                if lock_path.exists() and read_json_file(lock_path)!=selected:raise ValueError('Repeat selection lock changed')
                atomic_write_json(lock_path,selected)
                confirmed=execute(dataset+'_confirmation_seed'+str(seed),['-m','came_repro.protocol_repair','--stage','confirm_lp','--selection-lock',str(lock_path)])
                ranks=np.load(Path(confirmed['run_directory'])/'test_link_ranks.npz');metric=float((1/ranks['rank'].astype(np.float64)).mean())
                if abs(metric-confirmed['summary']['test']['official']['mrr'])>2e-7:raise ValueError('Repeat independent MRR mismatch')
                scores.append(metric*100);ledger['statistics'][dataset]={'completed':len(scores),'expected':10,'mean_mrr_pct':float(np.mean(scores)),'sample_sd_pp':float(np.std(scores,ddof=1)),'paper_mrr_pct':item['paper_mrr_pct'],'gap_pp':float(np.mean(scores)-item['paper_mrr_pct']),'scores_pct':scores,'all_seeds_included':True,'complete_repeat_grid':len(scores)==10};atomic_write_json(path,ledger)
        ledger['state']='completed_all_delivered_lp_repeats';atomic_write_json(path,ledger)
    finally:guard.close()


if __name__=='__main__':main()
