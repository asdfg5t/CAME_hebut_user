"""Frozen-encoder link prediction with official negatives and reverse-overlap reporting.

The paper does not release its link head: the chosen head and training recipe are
explicit reconstruction assumptions, not asserted author settings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .author_loader import ROOT, fingerprint
from .prepared_data import load_prepared
from .stream_inference import extract_full_embeddings
from .experiment import Experiment, seed_everything, write_json, atomic_checkpoint, rng_state
from .train import make_model
from .task_metrics import ranking_metrics, reverse_overlap_mask
from .pause_control import TrainingPaused, pause_requested


class LinkHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))

    def forward(self,source,target):
        return self.network(source*target).squeeze(-1)


def vectors(embeddings,index):
    return torch.from_numpy(np.array(embeddings[np.asarray(index,dtype=np.int64)],dtype=np.float32,copy=True)).cuda()


@torch.no_grad()
def evaluate_split(head,embeddings,split,batch_size,overlap,output_file=None):
    source,target,negative=(np.asarray(split[key]) for key in ('source_node','target_node','target_node_neg'))
    if negative.ndim!=2 or len(negative)!=len(source):
        raise ValueError('Official per-positive negative targets are required')
    all_sums={key:0. for key in ('mrr','hits@1','hits@3','hits@10')}
    clean_sums={key:0. for key in all_sums}
    clean_count=0
    ranks=[]
    positive_scores=[]
    head.eval()
    for start in range(0,len(source),batch_size):
        stop=min(start+batch_size,len(source))
        a=vectors(embeddings,source[start:stop])
        b=vectors(embeddings,target[start:stop])
        positive=head(a,b)
        negatives=vectors(embeddings,negative[start:stop].flatten()).reshape(stop-start,negative.shape[1],-1)
        negative_scores=head(a[:,None,:].expand_as(negatives),negatives)
        values=ranking_metrics(positive,negative_scores)
        if output_file is not None:
            ranks.append((1/values['mrr']).cpu().numpy())
            positive_scores.append(positive.cpu().numpy())
        keep=torch.from_numpy(~overlap[start:stop]).to('cuda')
        clean_count+=int(keep.sum())
        for key,value in values.items():
            all_sums[key]+=value.sum().item()
            clean_sums[key]+=value[keep].sum().item()
    if output_file is not None:
        np.savez_compressed(output_file,source_node=source,target_node=target,rank=np.concatenate(ranks),
                            positive_score=np.concatenate(positive_scores),reverse_overlap=overlap)
    return {'official':{key:value/len(source) for key,value in all_sums.items()},'queries':len(source),
            'reverse_overlap_queries':int(overlap.sum()),'nonoverlapping_queries':clean_count,
            'reverse_nonoverlap_subset':{key:value/clean_count for key,value in clean_sums.items()} if clean_count else None}


def train_link_head(embeddings,split,epochs,patience,seed,experiment,batch_size=4096,eval_batch_size=64,lr=.001,resume=None):
    seed_everything(seed)
    random=np.random.default_rng(seed)
    source,target=(np.asarray(split['train'][key],dtype=np.int64) for key in ('source_node','target_node'))
    n=len(embeddings)
    training_pairs=np.unique(source*n+target)
    overlaps={key:reverse_overlap_mask(source,target,np.asarray(value['source_node']),np.asarray(value['target_node']),n)
              for key,value in split.items() if key in ('valid','test')}
    np.savez_compressed(experiment.directory/'reverse_overlap_queries.npz',
                        **{key+'_excluded_query_indices':np.flatnonzero(value) for key,value in overlaps.items()})
    overlap_hash=fingerprint(experiment.directory/'reverse_overlap_queries.npz')
    head=LinkHead(embeddings.shape[1]).cuda()
    optimizer=torch.optim.Adam(head.parameters(),lr=lr)
    best,best_epoch,stale,best_state=-1.,0,0,None
    first_epoch=1
    history_rows=[]
    if resume:
        from .head_resume import restore_head
        previous=restore_head(resume,'came_link_head_v1',experiment,head,optimizer,random)
        first_epoch=previous['next_epoch']
        best,best_epoch,stale,best_state=previous['validation_mrr'],previous['selected_epoch'],previous['stale'],previous['best_state']
        history_rows=list(previous['history_rows'])
    epoch = first_epoch - 1
    for epoch in range(first_epoch,first_epoch if stale >= patience else epochs+1):
        head.train()
        order=random.permutation(len(source))
        total=0.
        for start in range(0,len(source),batch_size):
            index=order[start:start+batch_size]
            negative=random.integers(0,n,len(index),dtype=np.int64)
            def invalid_negatives():
                keys=source[index]*n+negative
                positions=np.searchsorted(training_pairs,keys)
                present=(positions<len(training_pairs)) & (training_pairs[np.minimum(positions,len(training_pairs)-1)]==keys)
                return (negative==source[index]) | present
            invalid=invalid_negatives()
            for _ in range(100):
                if not invalid.any():
                    break
                negative[invalid]=random.integers(0,n,int(invalid.sum()),dtype=np.int64)
                invalid=invalid_negatives()
            if invalid.any():
                raise RuntimeError('Unable to sample train-only nonedges')
            a,b,c=(vectors(embeddings,value) for value in (source[index],target[index],negative))
            optimizer.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(head(a,b),torch.ones(len(index),device='cuda')) + F.binary_cross_entropy_with_logits(head(a,c),torch.zeros(len(index),device='cuda'))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite link head loss')
            loss.backward()
            optimizer.step()
            total+=loss.item()*len(index)
        validation=evaluate_split(head,embeddings,split['valid'],eval_batch_size,overlaps['valid'])
        score=validation['official']['mrr']
        row={'step':epoch,'train_loss':total/len(source),'validation_mrr':score,'lr':lr}
        history_rows.append(row)
        experiment.metric(row)
        experiment.logger.info('LP epoch=%d loss=%.6f validation_MRR=%.6f',epoch,total/len(source),score)
        if score>best:
            best,best_epoch,stale=score,epoch,0
            best_state={key:value.detach().cpu().clone() for key,value in head.state_dict().items()}
        else:
            stale+=1
        atomic_checkpoint(experiment.checkpoint_directory/'link_head_last.pt',
            {'schema':'came_link_head_v1','head':head.state_dict(),'optimizer':optimizer.state_dict(),
             'rng':rng_state(),'sampling_generator':random.bit_generator.state,'completed_epochs':epoch,'next_epoch':epoch+1,
             'cursor_policy':'Epoch boundary; all train positives consumed','best_state':best_state,
             'selected_epoch':best_epoch,'validation_mrr':best,'stale':stale,
             'history_rows':history_rows,
             'config':experiment.config,'data_audit':experiment.data_audit})
        if pause_requested():
            raise TrainingPaused('LP head saved at epoch boundary; epoch=' + str(epoch))
        if stale>=patience:
            break
    head.load_state_dict(best_state)
    test=evaluate_split(head,embeddings,split['test'],eval_batch_size,overlaps['test'],experiment.directory/'test_link_ranks.npz')
    atomic_checkpoint(experiment.checkpoint_directory/'link_head_best.pt',{'head':best_state,'selected_epoch':best_epoch,'validation_mrr':best})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(line) for line in (experiment.directory/'logs/metrics.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([row['step'] for row in rows],[row['train_loss'] for row in rows],marker='o' if len(rows)==1 else None)
    axes[0].set(xlabel='LP head epoch',ylabel='Binary cross entropy',title='Frozen representation; reconstructed LP head')
    axes[1].plot([row['step'] for row in rows],[row['validation_mrr'] for row in rows],marker='o' if len(rows)==1 else None)
    axes[1].axvline(best_epoch,color='black',linestyle='--')
    axes[1].set(xlabel='LP head epoch',ylabel='Official validation MRR',title='Validation-only model selection')
    config=json.loads((experiment.directory/'config.json').read_text())
    fig.text(.02,.01,f'Source: {config["evaluation"]["dataset"]}; seed={seed}; official negatives; test used once after selection.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1))
    fig.savefig(experiment.directory/'figures/link_head_curves.png',dpi=180)
    plt.close(fig)
    return {'selected_epoch':best_epoch,'validation_mrr':best,'test':test,'test_evaluations':1,
            'reverse_overlap_indices_sha256':overlap_hash,'test_query_ranks_sha256':fingerprint(experiment.directory/'test_link_ranks.npz'),
            'head':'2-layer Hadamard-product MLP with ReLU','lr':lr,'epochs_executed':epoch,'max_epochs':epochs,'patience':patience,
            'negative_policy':'One uniform training-only nonedge per positive; official fixed candidates for validation/test',
            'assumptions':'Link head, optimizer/LR, train negatives and early-stopping patience reconstructed because author files/paper do not specify them.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--resume-head',type=Path)
    parser.add_argument('--representation',choices=['came','clip_text','clip_image'],default='came')
    parser.add_argument('--dataset',choices=['sports-copurchase','cloth-copurchase','books-lp'],required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--inference-batch-size',type=int,default=64)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    seed_everything(args.seed)
    storage=json.loads((ROOT/'configs/storage.json').read_text())
    if args.representation=='came':
        if args.checkpoint is None:
            raise ValueError('CAME requires a joint checkpoint')
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        config=saved['config']
        if config.get('datasets')!=['ogbn-products','books-lp','cloth-copurchase']:
            raise ValueError('LP evaluation requires the representation pretraining corpus')
        complete=saved.get('completed_epochs',saved.get('schedule',{}).get('epoch'))==8
    else:
        config={'stage':'frozen_clip_baseline','storage':storage,'deviations':['LP head/training recipe is reconstructed; no author LP trainer was provided.']}
        saved=None
        complete=None
    data=load_prepared(args.dataset,require_verified=True,build_graph=args.representation=='came')
    run_config={**config,'name':args.representation+'_lp_'+args.dataset.replace('-','_'),'seed':args.seed,
                'evaluation':{'task':'link','dataset':args.dataset,'epochs':args.epochs,'patience':args.patience,'seed':args.seed,
                              'representation':args.representation,'head':'hadamard_2layer_mlp','head_is_reconstructed':True,
                              'encoder_sha256':fingerprint(args.checkpoint) if args.checkpoint else None,
                              'resume_head':str(args.resume_head) if args.resume_head else None}}
    experiment=Experiment(run_config,data.audit,args.resume_head or args.checkpoint)
    try:
        if args.representation=='came':
            model=make_model(config)
            model.load_state_dict(saved['model'])
            del saved
            cache_root=Path(storage['embedding_cache_root'])
            cache=cache_root/fingerprint(args.checkpoint)[:20]/args.dataset
            embeddings=extract_full_embeddings(model,data,cache,fingerprint(args.checkpoint),args.inference_batch_size,experiment.logger)
            del model
            torch.cuda.empty_cache()
        else:
            embeddings=data.features[:,512:].numpy() if args.representation=='clip_image' else data.features[:,:512].numpy()
        raw=ROOT/'data/raw/mm_graph'/args.dataset/'lp-edge-split.pt'
        if fingerprint(raw)!=data.audit['raw_sha256']['lp-edge-split.pt']:
            raise ValueError('Official raw LP splits changed')
        split=torch.load(raw,map_location='cpu',weights_only=False)
        result=train_link_head(embeddings,split,args.epochs,args.patience,args.seed,experiment,resume=args.resume_head)
        result.update(task='link',dataset=args.dataset,representation=args.representation,seed=args.seed,pretraining_complete=complete,paper_reproduction_complete=False)
        write_json(experiment.directory/'summary.json',result)
        experiment.status('completed_downstream_task')
        print(f'RUN_DIR={experiment.directory}',flush=True)
    except BaseException as error:
        if isinstance(error, TrainingPaused):
            experiment.logger.info('Safe pause: %s', error)
        else:
            experiment.logger.exception('LP task failed')
        experiment.status('paused' if isinstance(error, TrainingPaused) else 'failed',error=str(error))
        raise
    finally:
        experiment.close()


if __name__=='__main__':
    main()
