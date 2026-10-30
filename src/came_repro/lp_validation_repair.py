"""Bounded validation-only frozen LP head trials informed by pinned MM-Graph code."""
import argparse
import copy
from pathlib import Path

import dgl
import numpy as np
import torch
from torch.nn import functional as F

from . import link_evaluation as link
from .atomic_io import read_json_file
from .author_loader import ROOT, fingerprint, verify_originals
from .experiment import Experiment, seed_everything, atomic_checkpoint, write_json, rng_state, restore_rng
from .head_resume import head_signature
from .pause_control import TrainingPaused, pause_requested
from .prepared_data import prepared_directory
from .remaining_lp_pilot import use_cached_vectors
from .task_metrics import reverse_overlap_mask


class LayerNormLinkHead(torch.nn.Module):
    def __init__(self, width=1024):
        super().__init__()
        self.network=torch.nn.Sequential(torch.nn.Linear(width,width),torch.nn.LayerNorm(width),torch.nn.ReLU(),
            torch.nn.Linear(width,width),torch.nn.LayerNorm(width),torch.nn.ReLU(),torch.nn.Linear(width,1))

    def forward(self,source,target):
        return self.network(source*target).squeeze(-1)


def make_head(variant):
    if variant=='existing_two_layer_head_upstream_schedule':return link.LinkHead(1024)
    if variant=='two_hidden_layernorm_head_upstream_schedule':return LayerNormLinkHead(1024)
    raise ValueError('Only the two declared LP candidates are allowed')


class TrainOnlyNegatives:
    def __init__(self,source,target,n):
        self.n=n
        self.keys=np.unique(source*n+target)

    def sample(self,source,random):
        negative=random.integers(0,self.n,len(source),dtype=np.int64)
        for _ in range(100):
            keys=source*self.n+negative;pos=np.searchsorted(self.keys,keys)
            invalid=(negative==source)|((pos<len(self.keys))&(self.keys[np.minimum(pos,len(self.keys)-1)]==keys))
            if not invalid.any():return negative
            negative[invalid]=random.integers(0,self.n,int(invalid.sum()),dtype=np.int64)
        raise RuntimeError('Unable to sample declared train-only nonedges')


def load_verified_cache(dataset,cache,model_hash):
    state=read_json_file(cache/'inference_state.json');file=cache/'fused_embeddings.npy'
    if (not state['complete'] or state['identity']['checkpoint_sha256']!=model_hash
        or state['identity']['data']['dataset']!=dataset or fingerprint(file)!=state['output_sha256']):
        raise ValueError('Complete fixed original LP cache required')
    matrix=np.load(file,mmap_mode='r')
    if matrix.shape!=(state['identity']['data']['nodes'],1024):raise ValueError('Full frozen 1024-wide matrix required')
    if matrix.nbytes+int(1.5*2**30)>torch.cuda.mem_get_info()[0]:raise RuntimeError('Exact CUDA cache exceeds scoring reserve')
    gpu=torch.empty(matrix.shape,dtype=torch.float32,device='cuda')
    for start in range(0,len(matrix),8192):
        stop=min(start+8192,len(matrix));gpu[start:stop]=torch.from_numpy(np.array(matrix[start:stop],copy=True)).cuda()
    return gpu,copy.deepcopy(state['identity']['data'])


def train_validation_only(gpu,train,valid,variant,experiment,resume=None):
    seed_everything(0);random=np.random.default_rng(0)
    source,target=(np.asarray(train[k],dtype=np.int64) for k in ('source_node','target_node'))
    sampler=TrainOnlyNegatives(source,target,len(gpu))
    overlap=reverse_overlap_mask(source,target,np.asarray(valid['source_node']),np.asarray(valid['target_node']),len(gpu))
    head=make_head(variant).cuda();optimizer=torch.optim.Adam(head.parameters(),lr=.0005)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=.1)
    best,best_epoch,stale,first,history,best_state=-1.,0,0,1,[],None
    if resume:
        old=torch.load(resume,map_location='cpu',weights_only=False)
        if (old['schema']!='came_lp_validation_repair_v1' or head_signature(old['config'])!=head_signature(experiment.config)
            or old['data_audit']!=experiment.data_audit):raise ValueError('LP repair resume changed protocol')
        head.load_state_dict(old['head']);optimizer.load_state_dict(old['optimizer']);scheduler.load_state_dict(old['scheduler'])
        restore_rng(old['rng']);random.bit_generator.state=old['sampling_generator']
        best,best_epoch,stale,first,history,best_state=old['best_mrr'],old['selected_epoch'],old['stale'],old['next_epoch'],old['history_rows'],old['best_state']
        for row in history:experiment.metric(row)
        experiment.logger.info('Restored complete LP repair epoch=%d; next=%d',first-1,first)
    for epoch in range(first,first if stale>=10 else 31):
        head.train();order=random.permutation(len(source));total=0.;used_lr=optimizer.param_groups[0]['lr']
        for start in range(0,len(source),512):
            ids=order[start:start+512];negative=sampler.sample(source[ids],random)
            a,b,c=(link.vectors(gpu,x) for x in (source[ids],target[ids],negative))
            optimizer.zero_grad(set_to_none=True)
            logits=torch.cat((head(a,b),head(a,c)));labels=torch.cat((torch.ones(len(ids),device='cuda'),torch.zeros(len(ids),device='cuda')))
            loss=F.binary_cross_entropy_with_logits(logits,labels)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite repaired LP objective')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('Nonfinite repaired LP gradient')
            optimizer.step();total+=loss.item()*len(ids)
        scheduler.step()
        metric=link.evaluate_split(head,gpu,valid,64,overlap)['official']['mrr']
        row={'step':epoch,'train_loss':total/len(source),'validation_mrr':metric,'lr_used':used_lr,'lr_next':optimizer.param_groups[0]['lr']}
        experiment.metric(row);history.append(row)
        if metric>best:
            best,best_epoch,stale=metric,epoch,0;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else:stale+=1
        experiment.logger.info('LP validation repair epoch=%d loss=%.6f val_MRR=%.6f lr_used=%.8g selected=%d',epoch,row['train_loss'],metric,used_lr,best_epoch)
        atomic_checkpoint(experiment.checkpoint_directory/'link_head_last.pt',{'schema':'came_lp_validation_repair_v1',
            'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(),
            'sampling_generator':random.bit_generator.state,'completed_epochs':epoch,'next_epoch':epoch+1,
            'cursor_policy':'full official train-positive epoch boundary','best_mrr':best,'selected_epoch':best_epoch,'stale':stale,
            'best_state':best_state,'history_rows':history,'config':experiment.config,'data_audit':experiment.data_audit})
        if pause_requested():raise TrainingPaused('LP validation repair saved epoch='+str(epoch))
        if stale>=10:break
    head.load_state_dict(best_state)
    metric=link.evaluate_split(head,gpu,valid,64,overlap,experiment.directory/'validation_link_ranks.npz')
    if abs(metric['official']['mrr']-best)>1e-12:raise ValueError('Best LP validation replay mismatch')
    atomic_checkpoint(experiment.checkpoint_directory/'link_head_best.pt',{'head':best_state,'selected_epoch':best_epoch,'validation_mrr':best})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['step'] for r in history],[r['train_loss'] for r in history]);axes[0].set(xlabel='Full train-positive epoch',ylabel='Binary cross entropy')
    axes[1].plot([r['step'] for r in history],[r['validation_mrr'] for r in history]);axes[1].axvline(best_epoch,ls='--',color='black')
    axes[1].set(xlabel='Full train-positive epoch',ylabel='Official validation MRR')
    fig.text(.02,.01,f'{experiment.config["evaluation"]["dataset"]}; seed=0; {variant}; fixed official negatives; test not evaluated.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(experiment.directory/'figures/lp_validation_repair_curves.png',dpi=180);plt.close(fig)
    return {'selected_epoch':best_epoch,'epochs_executed':len(history),'validation_mrr':best,'test':None,'test_evaluations':0,
        'head_variant':variant,'lr':.0005,'scheduler':'StepLR step5 gamma0.1','batch_size':512,'max_epochs':30,'patience':10}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['sports-copurchase','cloth-copurchase','books-lp'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--embedding-cache',type=Path,required=True)
    parser.add_argument('--variant',choices=['existing_two_layer_head_upstream_schedule','two_hidden_layernorm_head_upstream_schedule'],required=True)
    parser.add_argument('--resume-head',type=Path)
    args=parser.parse_args();verify_originals();seed_everything(0);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
    g=dgl.graph(([0],[1]),num_nodes=2).to('cuda');g.ndata['x']=torch.ones(2,1,device='cuda')
    g.update_all(dgl.function.copy_u('x','m'),dgl.function.sum('m','y'))
    if g.ndata['y'][1].item()!=1:raise RuntimeError('Actual DGL CUDA failed')
    del g
    model_hash=fingerprint(args.checkpoint);gpu,audit=load_verified_cache(args.dataset,args.embedding_cache,model_hash)
    manifest=read_json_file(prepared_directory(args.dataset)/'manifest.json')
    for name,digest in manifest['files'].items():
        if fingerprint(prepared_directory(args.dataset)/name)!=digest:raise ValueError('Official processed data changed')
    raw=ROOT/'data/raw/mm_graph'/args.dataset/'lp-edge-split.pt'
    if fingerprint(raw)!=audit['raw_sha256']['lp-edge-split.pt']:raise ValueError('Official LP split changed')
    split=torch.load(raw,map_location='cpu',weights_only=False)
    edges=np.load(prepared_directory(args.dataset)/'edge_index.npy',mmap_mode='r')
    np.testing.assert_array_equal(edges[:,0],np.asarray(split['train']['source_node']))
    np.testing.assert_array_equal(edges[:,1],np.asarray(split['train']['target_node']))
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);config=copy.deepcopy(saved['config']);del saved
    config.update(name='lp_repair_'+args.dataset+'_'+args.variant,seed=0,storage=read_json_file(ROOT/'configs/storage.json'))
    config['evaluation']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['evaluation'].update(task='link',seed=0,lr=.0005,batch_size=512,max_epochs=30,patience=10,
        scheduler_step=5,scheduler_gamma=.1,selection='official_validation',test_used_for_selection=False)
    config['deviations']=list(config.get('deviations',[]))+[
        'Frozen LP predictor and upstream-inspired optimization are bounded reconstruction candidates; CAME exact LP trainer remains unreleased.',
        'Encoder and predictor hidden width remain 1024; no new GNN/graph direction/official split modifications. Batch512/lr5e-4/StepLR5gamma0.1 use MM-Graph pinned defaults; predictor LayerNorm variant is separately declared.']
    experiment=Experiment(config,audit,args.resume_head or args.checkpoint)
    try:
        with use_cached_vectors():summary=train_validation_only(gpu,split['train'],split['valid'],args.variant,experiment,args.resume_head)
        summary.update(dataset=args.dataset,task='link',representation='came',seed=0,encoder_sha256=model_hash,
            diagnostic_only=True,paper_reproduction_complete=False,head_is_reconstruction=True)
        write_json(experiment.directory/'summary.json',summary);experiment.status('completed_lp_validation_repair')
    except BaseException as error:
        experiment.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
    finally:experiment.close()


if __name__=='__main__':main()
