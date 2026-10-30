"""Evidence-bound frozen Goodreads LP reconstruction; no simulated outcomes."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .atomic_io import atomic_write_json
from .author_loader import ROOT, fingerprint, verify_originals
from .evaluation_cursor import EvaluationCursor
from .experiment import Experiment, rng_state, restore_rng, seed_everything
from .goodreads_storage import atomic_checkpoint
from .pause_control import TrainingPaused, pause_requested
from .task_metrics import reverse_overlap_mask


def make_head(width=1024):
    """Pinned MM-Graph SAGE predictor, with CAME's fixed 1024-wide inputs."""
    return nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))


def pair_logits(head,source,target):
    return head(source*target).squeeze(-1)


def official_ranks(positive,negative):
    if negative.ndim!=2 or positive.shape!=(len(negative),):raise ValueError('Per-query score shape mismatch')
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():raise FloatingPointError('Nonfinite ranking scores')
    return 1.+.5*((negative>=positive[:,None]).sum(1)+(negative>positive[:,None]).sum(1))


def metrics(ranks):
    ranks=np.asarray(ranks,dtype=np.float64)
    if ranks.ndim!=1 or not len(ranks) or not np.isfinite(ranks).all() or not ((ranks>=1)&(ranks<=151)).all():
        raise ValueError('Complete finite official ranks required')
    return dict(mrr=float(np.mean(1./ranks)),**{'hits@1':float(np.mean(ranks<=1)),
        'hits@3':float(np.mean(ranks<=3)),'hits@10':float(np.mean(ranks<=10))})


def row_vectors(gpu,index):
    ids=torch.from_numpy(np.array(index,dtype=np.int64,copy=True)).to(gpu.device)
    return gpu[ids]


def require_finite_rows(matrix,block_rows=8192):
    """Exact all-row validation with bounded CUDA temporaries, never a subsample."""
    if block_rows<1:raise ValueError('Positive finite-check block size required')
    for start in range(0,len(matrix),block_rows):
        if pause_requested():raise TrainingPaused('Paused during full-cache integrity checks')
        if not torch.isfinite(matrix[start:start+block_rows]).all():
            raise FloatingPointError('Nonfinite frozen cache in rows%d:%d'%(start,min(start+block_rows,len(matrix))))


def resume_configuration_matches(saved,current):
    if saved==current:return True
    # One audited metadata-only transition: whole-matrix -> bounded finite check.
    if saved.get('evaluation',{}).get('release_sha256')!='9e8b21a27661a85f74c19d4e0312cb21230c2e6823206b5c4f2c1c541ba2293b':return False
    left,right=copy.deepcopy(saved),copy.deepcopy(current)
    for value in [left,right]:value.get('evaluation',{}).pop('release_sha256',None)
    return left==right


def evaluate(head,gpu,split,experiment=None,cursor=None,batch=64):
    source,target,negative=(np.asarray(split[k]) for k in ['source_node','target_node','target_node_neg'])
    n=len(source)
    if negative.shape!=(n,150):raise ValueError('All fixed official150 candidates required')
    if not n:raise ValueError('Empty evaluation split')
    head.eval();ranks=np.empty(n,dtype=np.float64) if cursor is None else None
    first=0 if cursor is None else cursor.cursor;last=time.monotonic();pending=[];commit_start=first
    with torch.no_grad():
        for start in range(first,n,batch):
            if pause_requested():raise TrainingPaused('Evaluation paused at committed query boundary')
            stop=min(start+batch,n);a=row_vectors(gpu,source[start:stop]);b=row_vectors(gpu,target[start:stop])
            c=row_vectors(gpu,negative[start:stop].reshape(-1)).reshape(stop-start,150,-1)
            values=official_ranks(pair_logits(head,a,b),pair_logits(head,a[:,None,:],c)).cpu().numpy().astype(np.float64)
            if cursor is None:ranks[start:stop]=values
            else:
                pending.append(values)
                if stop-commit_start>=1024 or stop==n:
                    cursor.commit(commit_start,np.concatenate(pending));commit_start=stop;pending=[]
            if experiment and (time.monotonic()-last>=30 or stop==n):
                experiment.logger.info('OFFICIAL_EVAL queries=%d/%d committed=%d fixed_negatives=150',stop,n,stop if cursor is None else cursor.cursor)
                last=time.monotonic()
    return ranks if cursor is None else cursor.complete_values()


def train_rounds(gpu,train,valid,recipe,seed,experiment,resume=None,pause_hook=None):
    """Sampled optimization rounds, NOT full official-training epochs."""
    seed_everything(seed);random=np.random.default_rng(seed)
    source,target=(np.asarray(train[k],dtype=np.int64) for k in ['source_node','target_node'])
    budget=min(recipe['positive_budget_per_round'],len(source));head=make_head(gpu.shape[1]).cuda()
    optimizer=torch.optim.Adam(head.parameters(),lr=recipe['lr'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,recipe['scheduler_step'],recipe['scheduler_gamma'])
    round_id,cursor,order,loss_sum=1,0,None,0.
    best,selected,stale,best_state,history=-1.,0,0,None,[]
    if resume:
        saved=torch.load(str(resume),map_location='cpu',weights_only=False)
        if saved['schema']!='came_goodreads_official_v1' or not resume_configuration_matches(saved['config'],experiment.config) or saved['data_audit']!=experiment.data_audit:
            raise ValueError('Goodreads resume protocol/data identity changed')
        head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng']);random.bit_generator.state=saved['sampling_generator']
        round_id,cursor,order,loss_sum=(saved[k] for k in ['next_round','positive_cursor','round_order','loss_sum'])
        best,selected,stale,best_state,history=(saved[k] for k in ['best_mrr','selected_round','stale','best_state','history_rows'])
        for row in history:experiment.metric(row)
        experiment.logger.info('RESUME round=%d positive_cursor=%d/%d completed_rounds=%d parent=%s',round_id,cursor,budget,len(history),resume)
    last_saved=time.monotonic()
    def save():
        nonlocal last_saved
        atomic_checkpoint(experiment.checkpoint_directory/'link_head_last.pt',dict(schema='came_goodreads_official_v1',
            head=head.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),
            sampling_generator=copy.deepcopy(random.bit_generator.state),next_round=round_id,positive_cursor=cursor,
            round_order=order,loss_sum=loss_sum,best_mrr=best,selected_round=selected,stale=stale,best_state=best_state,
            history_rows=history,config=experiment.config,data_audit=experiment.data_audit,
            positive_budget_per_round=budget,full_epoch_claimed=False))
        last_saved=time.monotonic()
    def boundary(phase):
        if pause_requested() or (pause_hook and pause_hook(round_id,cursor,phase)):
            save();raise TrainingPaused('Saved Goodreads complete optimizer state, round=%d cursor=%d'%(round_id,cursor))
    while round_id<=recipe['max_rounds'] and stale<recipe['patience']:
        if order is None:order=random.permutation(len(source))[:budget];cursor=0;loss_sum=0.
        head.train();started=time.monotonic();used_lr=optimizer.param_groups[0]['lr'];last_log=time.monotonic()
        for start in range(cursor,budget,recipe['batch_size']):
            ids=order[start:start+recipe['batch_size']]
            # Official DGL Uniform(1): target sampled uniformly; no val/test filtering.
            negatives=random.integers(0,len(gpu),len(ids),dtype=np.int64)
            a,b,c=(row_vectors(gpu,x) for x in [source[ids],target[ids],negatives])
            optimizer.zero_grad(set_to_none=True)
            logits=torch.cat([pair_logits(head,a,b),pair_logits(head,a,c)])
            labels=torch.cat([torch.ones(len(ids),device='cuda'),torch.zeros(len(ids),device='cuda')])
            loss=F.binary_cross_entropy_with_logits(logits,labels)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Goodreads training loss')
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('Nonfinite Goodreads gradient')
            optimizer.step();loss_sum+=loss.item()*len(ids);cursor=start+len(ids)
            boundary('train')
            if time.monotonic()-last_saved>=60:save()
            if time.monotonic()-last_log>=30:
                experiment.logger.info('TRAIN round=%d/%d positive_cursor=%d/%d pool=%d loss=%.7f CUDA_peak=%.1fMiB',round_id,recipe['max_rounds'],cursor,budget,len(source),loss_sum/cursor,torch.cuda.max_memory_allocated()/2**20)
                last_log=time.monotonic()
        save();boundary('validation');train_seconds=time.monotonic()-started;vstart=time.monotonic()
        ranks=evaluate(head,gpu,valid,experiment);score=metrics(ranks)['mrr']
        row=dict(step=round_id,train_loss=loss_sum/budget,validation_mrr=score,lr=used_lr,
            positives_this_round=budget,completed_positive_presentations=round_id*budget,full_official_epochs_claimed=0)
        history.append(row);experiment.metric(row)
        if score>best:
            best,selected,stale=score,round_id,0;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            atomic_checkpoint(experiment.checkpoint_directory/'link_head_best.pt',dict(head=best_state,selected_round=selected,validation_mrr=best))
            np.savez_compressed(experiment.directory/'validation_link_ranks.npz',source_node=np.asarray(valid['source_node']),target_node=np.asarray(valid['target_node']),rank=ranks)
        else:stale+=1
        experiment.logger.info('ROUND_COMPLETE round=%d val_MRR=%.8f best=%.8f selected=%d train=%.2fs validation=%.2fs; budget rounds, not full epochs',round_id,score,best,selected,train_seconds,time.monotonic()-vstart)
        with (experiment.directory/'logs/timing.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(round=round_id,train_seconds=train_seconds,validation_seconds=time.monotonic()-vstart,peak_cuda_bytes=torch.cuda.max_memory_allocated()))+'\n')
        scheduler.step();round_id+=1;cursor=0;order=None;loss_sum=0.;save();boundary('round_end')
    if best_state is None:raise ValueError('No complete validation-selected head')
    head.load_state_dict(best_state)
    replay=evaluate(head,gpu,valid,experiment)
    if metrics(replay)['mrr']!=best:raise ValueError('Best validation replay changed')
    np.savez_compressed(experiment.directory/'validation_link_ranks.npz',source_node=np.asarray(valid['source_node']),target_node=np.asarray(valid['target_node']),rank=replay)
    atomic_checkpoint(experiment.checkpoint_directory/'link_head_best.pt',dict(head=best_state,selected_round=selected,validation_mrr=best))
    draw_curves(history,selected,experiment)
    return dict(dataset='books-lp',task='link',seed=seed,validation_mrr=best,selected_round=selected,
        rounds_completed=len(history),test_evaluations=0,test=None,head_variant='pinned_sage_predictor_1024',
        recipe=recipe,official_train_pool=len(source),positive_budget_per_round=budget,
        full_official_epochs_claimed=False,paper_reproduction_complete=False,head_is_reconstruction=True)


def draw_curves(history,selected,experiment):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['step'] for r in history],[r['train_loss'] for r in history]);axes[0].set(xlabel='Optimization round (512000 positives maximum)',ylabel='BCE loss',title='Frozen encoder: train objective')
    axes[1].plot([r['step'] for r in history],[100*r['validation_mrr'] for r in history]);axes[1].axvline(selected,color='black',ls='--')
    axes[1].set(xlabel='Optimization round',ylabel='Complete official validation MRR (%)',title='Validation-only selection')
    label='Synthetic fixture ONLY; not a paper score' if experiment.data_audit.get('fixture_only') else 'Goodreads-LP; 150 official fixed negatives'
    fig.text(.02,.01,label+'; seed%d; budget rounds are NOT full epochs; test held out.'%experiment.config['seed'],fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(experiment.directory/'figures/goodreads_training.png',dpi=180);plt.close(fig)


def load_data(profile):
    from .lp_validation_repair import load_verified_cache
    from .prepared_data import prepared_directory
    verify_originals();raw=ROOT/'data/raw/mm_graph/books-lp/lp-edge-split.pt'
    if fingerprint(raw)!=profile['split_sha256']:raise ValueError('Official raw split identity changed')
    gpu,audit=load_verified_cache('books-lp',Path(profile['embedding_cache']),profile['encoder_sha256'])
    split=torch.load(str(raw),map_location='cpu',weights_only=False,mmap=True)
    for name,n in [('train',3437017),('valid',570464),('test',1710677)]:
        if len(split[name]['source_node'])!=n or len(split[name]['target_node'])!=n:raise ValueError('Full official split required')
        if name!='train' and np.asarray(split[name]['target_node_neg']).shape!=(n,150):raise ValueError('Fixed official150 negatives required')
    directory=prepared_directory('books-lp')
    if fingerprint(directory/'manifest.json')!=profile['prepared_manifest_sha256']:raise ValueError('Official prepared manifest changed')
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        if fingerprint(directory/name)!=digest:raise ValueError('Official prepared asset changed: '+name)
    edges=np.load(directory/'edge_index.npy',mmap_mode='r')
    np.testing.assert_array_equal(edges[:,0],split['train']['source_node']);np.testing.assert_array_equal(edges[:,1],split['train']['target_node'])
    if tuple(gpu.shape)!=(636502,1024):raise ValueError('Full636502x1024 CUDA cache required')
    require_finite_rows(gpu)
    return gpu,split,audit


def test_saved(profile,lock_file,resume=None):
    lock=json.loads(Path(lock_file).read_text(encoding='utf-8'));run=Path(lock['validation_run_directory'])
    s=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if s['test_evaluations']!=0 or lock['test_used_for_selection'] or not lock['locked_before_test']:raise ValueError('Validation-only selected saved head required')
    if fingerprint(run/'summary.json')!=lock['validation_summary_sha256'] or fingerprint(lock['classifier_checkpoint'])!=lock['classifier_sha256']:raise ValueError('Selected head identity changed')
    gpu,split,audit=load_data(profile);config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    config=copy.deepcopy(config);config['name']='goodreads_official_test';config['evaluation']['selection_lock_sha256']=fingerprint(lock_file)
    e=Experiment(config,audit,Path(resume) if resume else Path(lock['classifier_checkpoint']))
    try:
        head=make_head().cuda();head.load_state_dict(torch.load(lock['classifier_checkpoint'],map_location='cpu',weights_only=False)['head'])
        replay=evaluate(head,gpu,split['valid'],e)
        with np.load(run/'validation_link_ranks.npz') as old:np.testing.assert_array_equal(replay,old['rank'])
        if metrics(replay)['mrr']!=s['validation_mrr']:raise ValueError('Saved validation replay changed')
        cursor=EvaluationCursor(e.checkpoint_directory/'test_cursor.json',dict(lock_sha256=fingerprint(lock_file),canonical_block=64),1710677,'float64',resume)
        ranks=evaluate(head,gpu,split['test'],e,cursor)
        overlap=reverse_overlap_mask(np.asarray(split['train']['source_node']),np.asarray(split['train']['target_node']),np.asarray(split['test']['source_node']),np.asarray(split['test']['target_node']),636502)
        if int(overlap.sum())!=108818:raise ValueError('Official reverse overlap changed')
        np.savez_compressed(e.directory/'test_link_ranks.npz',source_node=np.asarray(split['test']['source_node']),target_node=np.asarray(split['test']['target_node']),rank=ranks,reverse_overlap=overlap)
        s=copy.deepcopy(s);s.update(test=dict(official=metrics(ranks),queries=len(ranks),reverse_overlap_queries=int(overlap.sum()),reverse_nonoverlap_subset=metrics(ranks[~overlap])),test_evaluations=1,validation_run_directory=str(run),selection_lock_sha256=fingerprint(lock_file),deterministic_test_cursor_complete=True)
        atomic_write_json(e.directory/'summary.json',s)
        e.metric(dict(step=len(ranks),test_mrr=s['test']['official']['mrr'],official_queries=len(ranks),test_evaluations=1))
        e.logger.info('COMPLETE_OFFICIAL_TEST MRR=%.8f%% paper=10.42%% gap=%+.8fpp queries=%d; no filtering, no strategy search',s['test']['official']['mrr']*100,s['test']['official']['mrr']*100-10.42,len(ranks))
        e.status('completed_goodreads_official_test')
    except BaseException as error:e.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
    finally:e.close()
