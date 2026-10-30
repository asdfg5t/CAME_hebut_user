"""One declared sampled-listwise LP objective; official validation/test unchanged."""

def train_listwise(gpu,train,valid,recipe,seed,experiment,resume=None):
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
        if saved['schema']!='came_listwise_lp_v1' or saved['config']!=experiment.config or saved['data_audit']!=experiment.data_audit or saved['seed']!=seed:raise ValueError('Listwise LP resume identity changed')
        head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer'])
        if scheduler:scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng']);random.bit_generator.state=saved['sampling_generator']
        best,selected,stale,first,history,best_state=saved['best_mrr'],saved['selected_epoch'],saved['stale'],saved['next_epoch'],saved['history_rows'],saved['best_state']
        for row in history:experiment.metric(row)
    for epoch in range(first,first if stale>=recipe['patience'] else recipe['max_epochs']+1):
        head.train();order=random.permutation(len(source));total=0.;hard_total=0.;lr=opt.param_groups[0]['lr']
        for start in range(0,len(order),recipe['batch_size']):
            ids=order[start:start+recipe['batch_size']]
            pool=sampler.sample(np.repeat(source[ids],4),random).reshape(len(ids),4)
            candidates=np.concatenate((target[ids,None],pool),axis=1)
            a=link.vectors(gpu,source[ids])[:,None,:]
            b=link.vectors(gpu,candidates.ravel()).reshape(len(ids),5,-1)
            opt.zero_grad(set_to_none=True);scores=head(a.expand_as(b),b)
            hard_total+=float(scores[:,1:].detach().max(1).values.mean())*len(ids)
            loss=listwise_loss(scores)
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
        experiment.logger.info('Listwise LP seed=%d epoch=%d listCE=%.6f validation_MRR=%.8f selected=%d',seed,epoch,row['train_loss'],value,selected)
        atomic_checkpoint(experiment.checkpoint_directory/'link_head_last.pt',{'schema':'came_listwise_lp_v1','seed':seed,'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,'rng':rng_state(),'sampling_generator':random.bit_generator.state,'completed_epochs':epoch,'next_epoch':epoch+1,'best_mrr':best,'selected_epoch':selected,'stale':stale,'best_state':best_state,'history_rows':history,'config':experiment.config,'data_audit':experiment.data_audit,'cursor_policy':'full official train-positive epoch boundary','negative_sampling':'4 uniform train-only nonedges jointly ranked; train and reverse train positives excluded'})
        if pause_requested():raise TrainingPaused('Listwise LP complete epoch checkpoint saved')
        if stale>=recipe['patience']:break
    head.load_state_dict(best_state);replay=link.evaluate_split(head,gpu,valid,64,overlap,experiment.directory/'validation_link_ranks.npz')
    if abs(replay['official']['mrr']-best)>1e-12:raise ValueError('Selected hard LP validation replay mismatch')
    atomic_checkpoint(experiment.checkpoint_directory/'link_head_best.pt',{'head':best_state,'selected_epoch':selected,'validation_mrr':best})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));steps=[r['step'] for r in history]
    axes[0].plot(steps,[r['train_loss'] for r in history]);axes[0].set(xlabel='Full train-positive epoch',ylabel='Sampled listwise cross entropy')
    axes[1].plot(steps,[r['validation_mrr'] for r in history]);axes[1].axvline(selected,ls='--',color='black');axes[1].set(xlabel='Full train-positive epoch',ylabel='Official validation MRR')
    fig.text(.02,.01,'Training dropout0.3; Seed%d; positive plus4 uniform train-only nonedges; listwise CE; no val/test labels for training; test held out.'%seed,fontsize=8);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(experiment.directory/'figures/hard_lp_curves.png',dpi=180);plt.close(fig)
    return dict(recipe,seed=seed,selected_epoch=selected,epochs_executed=len(history),validation_mrr=best,test=None,test_evaluations=0,negative_pool_size=4,negative_policy='four_uniform_train_and_reverse_known_nonedges',loss_reduction='listwise_cross_entropy')


def listwise_loss(scores):
    import torch
    from torch.nn import functional as F
    if scores.ndim != 2 or scores.shape[1] != 5:
        raise ValueError('Positive column0 and exactly four training negatives required')
    return F.cross_entropy(scores, torch.zeros(len(scores),dtype=torch.long,device=scores.device))
