"""Independent frozen-trainer adaptation: add training dropout; eval function unchanged."""

def train_regularized(gpu,train,valid,recipe,seed,experiment,resume=None):
    from run_lp_hard_negative_candidate import sample_hard_negative
    from came_repro.regularized_link_head import RegularizedLinkHead
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
    head=RegularizedLinkHead(dropout=recipe['head_training_dropout']).cuda();opt=torch.optim.Adam(head.parameters(),lr=recipe['lr'])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,recipe['step_lr_size'],recipe['step_lr_gamma']) if recipe['step_lr_size'] else None
    best,selected,stale,first,history,best_state=-1.,0,0,1,[],None
    if resume:
        saved=torch.load(str(resume),map_location='cpu',weights_only=False)
        if saved['schema']!='came_regularized_lp_v1' or saved['config']!=experiment.config or saved['data_audit']!=experiment.data_audit or saved['seed']!=seed:raise ValueError('Regularized LP resume identity changed')
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
        experiment.logger.info('Regularized LP seed=%d epoch=%d BCE=%.6f validation_MRR=%.8f selected=%d',seed,epoch,row['train_loss'],value,selected)
        atomic_checkpoint(experiment.checkpoint_directory/'link_head_last.pt',{'schema':'came_regularized_lp_v1','seed':seed,'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,'rng':rng_state(),'sampling_generator':random.bit_generator.state,'completed_epochs':epoch,'next_epoch':epoch+1,'best_mrr':best,'selected_epoch':selected,'stale':stale,'best_state':best_state,'history_rows':history,'config':experiment.config,'data_audit':experiment.data_audit,'cursor_policy':'full official train-positive epoch boundary','negative_sampling':'max of4 uniform nonedges; train and reverse train positives excluded'})
        if pause_requested():raise TrainingPaused('Regularized LP complete epoch checkpoint saved')
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
    fig.text(.02,.01,'Training dropout0.3; Seed%d; max-scoring of4 uniform train-only nonedges; no val/test labels for training; test held out.'%seed,fontsize=8);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(experiment.directory/'figures/hard_lp_curves.png',dpi=180);plt.close(fig)
    return dict(recipe,seed=seed,selected_epoch=selected,epochs_executed=len(history),validation_mrr=best,test=None,test_evaluations=0,negative_pool_size=4,negative_policy='max_score_train_and_reverse_known_nonedge')
