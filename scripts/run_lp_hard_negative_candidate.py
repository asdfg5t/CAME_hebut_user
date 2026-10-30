"""One declared four-way train-only hard-negative candidate, retaining frozen CAME/LP head."""
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


def sample_hard_negative(head,gpu,source,sampler,random):
    import numpy as np
    import torch
    from came_repro import link_evaluation as link
    pool=sampler.sample(np.repeat(source,4),random).reshape(len(source),4)
    a=link.vectors(gpu,source);b=link.vectors(gpu,pool.ravel()).reshape(len(source),4,-1)
    mode=head.training;head.eval()
    with torch.no_grad():scores=head(a[:,None,:].expand_as(b),b);selected=scores.argmax(1).cpu().numpy()
    head.train(mode)
    return pool[np.arange(len(source)),selected],pool,scores.detach()


def train_hard(gpu,train,valid,recipe,seed,experiment,resume=None):
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
    sampler=TrainOnlyNegatives(np.concatenate((source,target)),np.concatenate((target,source)),len(gpu))
    overlap=reverse_overlap_mask(source,target,np.asarray(valid['source_node']),np.asarray(valid['target_node']),len(gpu))
    head=make_head(recipe['head_variant']).cuda();opt=torch.optim.Adam(head.parameters(),lr=recipe['lr'])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,recipe['step_lr_size'],recipe['step_lr_gamma']) if recipe['step_lr_size'] else None
    best,selected,stale,first,history,best_state=-1.,0,0,1,[],None
    if resume:
        saved=torch.load(str(resume),map_location='cpu',weights_only=False)
        if saved['schema']!='came_hard_lp_candidate_v1' or saved['config']!=experiment.config or saved['data_audit']!=experiment.data_audit or saved['seed']!=seed:raise ValueError('Hard LP resume identity changed')
        head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer'])
        if scheduler:scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng']);random.bit_generator.state=saved['sampling_generator']
        best,selected,stale,first,history,best_state=saved['best_mrr'],saved['selected_epoch'],saved['stale'],saved['next_epoch'],saved['history_rows'],saved['best_state']
        for row in history:experiment.metric(row)
    for epoch in range(first,first if stale>=recipe['patience'] else recipe['max_epochs']+1):
        head.train();order=random.permutation(len(source));total=0.;hard_total=0.;lr=opt.param_groups[0]['lr']
        for start in range(0,len(order),recipe['batch_size']):
            ids=order[start:start+recipe['batch_size']];negative,pool,scores=sample_hard_negative(head,gpu,source[ids],sampler,random)
            hard_total+=float(scores.max(1).values.mean())*len(ids)
            a,b,c=(link.vectors(gpu,x) for x in (source[ids],target[ids],negative))
            opt.zero_grad(set_to_none=True);positive,neg=head(a,b),head(a,c)
            if recipe['loss_reduction']=='separate_mean_sum':loss=F.binary_cross_entropy_with_logits(positive,torch.ones_like(positive))+F.binary_cross_entropy_with_logits(neg,torch.zeros_like(neg))
            else:loss=F.binary_cross_entropy_with_logits(torch.cat((positive,neg)),torch.cat((torch.ones_like(positive),torch.zeros_like(neg))))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite hard LP loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('Nonfinite hard LP gradients')
            opt.step();total+=loss.item()*len(ids)
        if scheduler:scheduler.step()
        value=link.evaluate_split(head,gpu,valid,64,overlap)['official']['mrr']
        row={'step':epoch,'train_loss':total/len(source),'validation_mrr':value,'lr_used':lr,'lr_next':opt.param_groups[0]['lr'],'mean_selected_negative_score':hard_total/len(source),'negative_pool_size':4}
        history.append(row);experiment.metric(row)
        if value>best:
            best,selected,stale=value,epoch,0;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else:stale+=1
        experiment.logger.info('Hard LP seed=%d epoch=%d BCE=%.6f validation_MRR=%.8f selected=%d',seed,epoch,row['train_loss'],value,selected)
        atomic_checkpoint(experiment.checkpoint_directory/'link_head_last.pt',{'schema':'came_hard_lp_candidate_v1','seed':seed,'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,'rng':rng_state(),'sampling_generator':random.bit_generator.state,'completed_epochs':epoch,'next_epoch':epoch+1,'best_mrr':best,'selected_epoch':selected,'stale':stale,'best_state':best_state,'history_rows':history,'config':experiment.config,'data_audit':experiment.data_audit,'cursor_policy':'full official train-positive epoch boundary','negative_sampling':'max of4 uniform nonedges; train and reverse train positives excluded'})
        if pause_requested():raise TrainingPaused('Hard LP complete epoch checkpoint saved')
        if stale>=recipe['patience']:break
    head.load_state_dict(best_state);replay=link.evaluate_split(head,gpu,valid,64,overlap,experiment.directory/'validation_link_ranks.npz')
    if abs(replay['official']['mrr']-best)>1e-12:raise ValueError('Selected hard LP validation replay mismatch')
    atomic_checkpoint(experiment.checkpoint_directory/'link_head_best.pt',{'head':best_state,'selected_epoch':selected,'validation_mrr':best})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));steps=[r['step'] for r in history]
    axes[0].plot(steps,[r['train_loss'] for r in history]);axes[0].set(xlabel='Full train-positive epoch',ylabel='Binary cross entropy')
    axes[1].plot(steps,[r['validation_mrr'] for r in history]);axes[1].axvline(selected,ls='--',color='black');axes[1].set(xlabel='Full train-positive epoch',ylabel='Official validation MRR')
    fig.text(.02,.01,'Seed%d; max-scoring of4 uniform train-only nonedges; no val/test labels for training; test held out.'%seed,fontsize=8);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(experiment.directory/'figures/hard_lp_curves.png',dpi=180);plt.close(fig)
    return dict(recipe,seed=seed,selected_epoch=selected,epochs_executed=len(history),validation_mrr=best,test=None,test_evaluations=0,negative_pool_size=4,negative_policy='max_score_train_and_reverse_known_nonedge')


def main():
    import json,numpy as np,torch
    from came_repro.atomic_io import read_json_file
    from came_repro.author_loader import fingerprint,verify_originals
    from came_repro.experiment import Experiment,write_json,seed_everything
    from came_repro.protocol_repair import check_cuda
    from came_repro.lp_validation_repair import TrainOnlyNegatives,make_head,load_verified_cache
    from came_repro.remaining_lp_pilot import use_cached_vectors
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',choices=['sports-copurchase','cloth-copurchase']);parser.add_argument('--reference-run',type=Path);parser.add_argument('--seed',type=int,default=0);parser.add_argument('--resume-head',type=Path);parser.add_argument('--acceptance',action='store_true');args=parser.parse_args()
    verify_originals();check_cuda();torch.set_num_threads(4)
    profile_path=ROOT/'configs/lp_negative_objective_20261014.json';profile=read_json_file(profile_path)
    if args.acceptance:
        seed_everything(218);gpu=torch.randn(12,1024,device='cuda');source=np.array([0,1,2,3,4,5]);target=source+1
        sampler=TrainOnlyNegatives(np.concatenate((source,target)),np.concatenate((target,source)),12);head=make_head('existing_two_layer_head_upstream_schedule').cuda()
        with use_cached_vectors():negative,pool,scores=sample_hard_negative(head,gpu,source,sampler,np.random.default_rng(0))
        np.testing.assert_array_equal(negative,pool[np.arange(len(source)),scores.argmax(1).cpu().numpy()])
        keys=np.unique(np.concatenate((source*12+target,target*12+source)))
        assert not np.isin(np.repeat(source,4)*12+pool.ravel(),keys).any() and not (pool==source[:,None]).any()
        train={'source_node':source,'target_node':target};valid={'source_node':np.array([6,7]),'target_node':np.array([7,8]),'target_node_neg':np.tile(np.arange(12),(2,13))[:,:150]}
        recipe={'head_variant':'existing_two_layer_head_upstream_schedule','lr':.001,'batch_size':3,'max_epochs':3,'patience':10,'step_lr_size':None,'step_lr_gamma':None,'loss_reduction':'separate_mean_sum'}
        config={'name':'hard_lp_resume_acceptance','seed':1,'storage':read_json_file(ROOT/'configs/storage.json'),'evaluation':{'fixture_only':True}}
        baseline=Experiment(config,{'fixture_only':True})
        with use_cached_vectors():train_hard(gpu,train,valid,recipe,1,baseline)
        baseline.status('completed_acceptance_fixture');baseline.close()
        import came_repro.pause_control as pauses
        original=pauses.pause_requested;count=[0]
        def pause_at_two():count[0]+=1;return count[0]==2
        paused=Experiment(config,{'fixture_only':True});pauses.pause_requested=pause_at_two
        try:
            try:
                with use_cached_vectors():train_hard(gpu,train,valid,recipe,1,paused)
            except pauses.TrainingPaused:paused.status('paused_acceptance_fixture')
            else:raise ValueError('Hard LP cooperative pause fixture did not pause')
        finally:pauses.pause_requested=original;paused.close()
        cp=paused.checkpoint_directory/'link_head_last.pt';resumed=Experiment(config,{'fixture_only':True},cp)
        with use_cached_vectors():train_hard(gpu,train,valid,recipe,1,resumed,cp)
        resumed.status('completed_acceptance_fixture');resumed.close()
        a=torch.load(str(baseline.checkpoint_directory/'link_head_last.pt'),map_location='cpu',weights_only=False);b=torch.load(str(resumed.checkpoint_directory/'link_head_last.pt'),map_location='cpu',weights_only=False)
        assert a['history_rows']==b['history_rows'] and a['next_epoch']==b['next_epoch']==4
        for name in a['head']:assert torch.equal(a['head'][name],b['head'][name])
        report={'passed':True,'actual_Torch_DGL_CUDA':True,'hard_argmax_correct':True,'known_train_and_reverse_nonedge_checked':True,'seed1_pause_resume_bitexact_weights_history':True,'profile_sha256':fingerprint(profile_path),'script_sha256':fingerprint(__file__),'resume_parent_sha256':fingerprint(cp),'fixture_runs':[str(e.directory) for e in [baseline,paused,resumed]]}
        write_json(ROOT/'artifacts/audit/hard_lp_CUDA_acceptance_20261014.json',report);print(json.dumps(report));return
    accepted=read_json_file(ROOT/'artifacts/audit/hard_lp_CUDA_acceptance_20261014.json')
    if not accepted['passed'] or accepted['script_sha256']!=fingerprint(__file__) or accepted['profile_sha256']!=fingerprint(profile_path):raise ValueError('Frozen hard LP CUDA/resume acceptance required')
    reference=read_json_file(args.reference_run/'summary.json');digest=reference['encoder_sha256'];cache=Path(reference['embedding_cache'])
    gpu,audit=load_verified_cache(args.dataset,cache,digest);config=read_json_file(args.reference_run/'config.json')
    recipe={k:reference[k] for k in ['head_variant','lr','batch_size','max_epochs','patience','step_lr_size','step_lr_gamma','loss_reduction']}
    recipe['max_epochs']=profile['budget']['sports_max_epochs' if args.dataset=='sports-copurchase' else 'cloth_max_epochs']
    config.update(name='hard_lp_'+args.dataset,seed=args.seed);config['evaluation'].update(recipe,seed=args.seed,negative_pool_size=4,negative_policy='max_score_train_and_reverse_known_nonedge',hard_profile_sha256=fingerprint(profile_path),hard_script_sha256=fingerprint(__file__),test_used_for_selection=False)
    encoder=Path(read_json_file(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint']);experiment=Experiment(config,audit,args.resume_head or encoder)
    try:
        split=torch.load(str(ROOT/'data/raw/mm_graph'/args.dataset/'lp-edge-split.pt'),map_location='cpu',weights_only=False,mmap=True)
        with use_cached_vectors():summary=train_hard(gpu,split['train'],split['valid'],recipe,args.seed,experiment,args.resume_head)
        summary.update(dataset=args.dataset,task='link',representation='came',encoder_sha256=digest,embedding_cache=str(cache),graph_policy='upstream_train_bidirected_unique',diagnostic_only=args.seed==0,paper_reproduction_complete=False)
        write_json(experiment.directory/'summary.json',summary);experiment.status('completed_hard_lp_validation')
    except BaseException as error:
        from came_repro.pause_control import TrainingPaused
        experiment.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
    finally:experiment.close()


if __name__=='__main__':main()
