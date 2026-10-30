"""Validation-only fixed-step convex optimization of the paper's frozen KG linear head."""
import argparse
import copy
import hashlib
from pathlib import Path

import dgl
import numpy as np
import torch
from torch.nn import functional as F

from .atomic_io import read_json_file
from .author_loader import ROOT, fingerprint, verify_originals
from .downstream import score_probe
from .experiment import Experiment, seed_everything, atomic_checkpoint, write_json, rng_state, restore_rng
from .feature_affine import training_pair_statistics, fold_affine_linear
from .head_resume import head_signature
from .kg_affine_probe import AffinePairInput, DoublePairInput
from .kg_graph_probe import kg_input
from .pause_control import TrainingPaused, pause_requested
from .prepared_data import prepared_directory


def make_solver(parameters, lr=.05):
    return torch.optim.LBFGS(parameters, lr=lr, max_iter=1, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn=None)


def linear_objective(classifier, features, labels, l2):
    cross_entropy = F.cross_entropy(classifier(features), labels)
    penalty = .5*l2*classifier.weight.square().sum()
    return cross_entropy+penalty, cross_entropy, penalty


def train_candidate(train, valid, classes, l2, experiment, resume=None):
    seed_everything(0)
    head = torch.nn.Linear(train.width, classes).cuda()
    optimizer = make_solver(head.parameters())
    features, labels = train.batch(np.arange(len(train)))
    best_state, best_epoch, best_accuracy, stale, first, rows = None, 0, -1., 0, 1, []
    if resume:
        previous = torch.load(resume, map_location='cpu', weights_only=False)
        if (previous['schema'] != 'came_convex_linear_probe_v1'
                or head_signature(previous['config']) != head_signature(experiment.config)
                or previous['data_audit'] != experiment.data_audit):
            raise ValueError('Convex probe resume changed data/model/protocol')
        head.load_state_dict(previous['classifier']);optimizer.load_state_dict(previous['optimizer'])
        restore_rng(previous['rng'])
        first, best_epoch, best_accuracy, stale = previous['next_epoch'], previous['selected_epoch'], previous['best_accuracy'], previous['stale']
        best_state, rows = previous['best_state'], previous['history_rows']
        for row in rows: experiment.metric(row)
        experiment.logger.info('Restored complete convex epoch=%d; next=%d', first-1, first)
    epoch = first-1
    for epoch in range(first, first if stale >= 100 else 2001):
        head.train(); diagnostics = {}
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, ce, penalty = linear_objective(head, features, labels, l2)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite convex objective')
            loss.backward()
            grads = [p.grad for p in head.parameters()]
            if not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError('Nonfinite convex gradient')
            diagnostics.update(train_loss=ce.item(), l2_penalty=penalty.item(),
                gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in grads))))
            return loss
        optimizer.step(closure)
        accuracy, loss, _ = score_probe(head, valid)
        row = {'step':epoch, **diagnostics, 'val_loss':loss, 'val_accuracy':accuracy, 'lr':.05}
        experiment.metric(row);rows.append(row)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch, stale = accuracy, epoch, 0
            best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else: stale += 1
        if epoch == 1 or epoch % 25 == 0:
            experiment.logger.info('convex epoch=%d CE=%.6f validation_accuracy=%.6f selected=%d',epoch,diagnostics['train_loss'],accuracy,best_epoch)
        requested = pause_requested()
        if epoch == 1 or epoch % 100 == 0 or stale >= 100 or epoch == 2000 or requested:
            atomic_checkpoint(experiment.checkpoint_directory/'probe_last.pt',
                {'schema':'came_convex_linear_probe_v1','classifier':head.state_dict(),'optimizer':optimizer.state_dict(),
                 'rng':rng_state(),'completed_epochs':epoch,'next_epoch':epoch+1,'cursor_policy':'full official train-pair epoch boundary',
                 'best_state':best_state,'selected_epoch':best_epoch,'best_accuracy':best_accuracy,'stale':stale,
                 'history_rows':rows,'config':experiment.config,'data_audit':experiment.data_audit})
        if requested: raise TrainingPaused('Convex linear probe saved epoch='+str(epoch))
        if stale >= 100: break
    head.load_state_dict(best_state)
    accuracy, loss, predictions = score_probe(head, valid, save_predictions=True)
    if accuracy != best_accuracy: raise ValueError('Selected validation model mismatch')
    atomic_checkpoint(experiment.checkpoint_directory/'probe_best.pt',
        {'classifier':best_state,'selected_epoch':best_epoch,'validation_accuracy':accuracy})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['step'] for r in rows],[r['train_loss'] for r in rows],label='train CE')
    axes[0].plot([r['step'] for r in rows],[r['val_loss'] for r in rows],label='validation CE');axes[0].legend()
    axes[0].set(xlabel='Full train-pair epoch',ylabel='Cross entropy',title='Frozen concatenation linear classifier')
    axes[1].plot([r['step'] for r in rows],[r['val_accuracy'] for r in rows]);axes[1].axvline(best_epoch,ls='--',color='black')
    axes[1].set(xlabel='Full train-pair epoch',ylabel='Validation accuracy',title='Validation selection; test held out')
    fig.text(.02,.01,f'WN18RR seed=0; fixed-step LBFGS lr=.05; L2={l2}; full official validation; engineering optimizer.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(experiment.directory/'figures/convex_probe_curves.png',dpi=180);plt.close(fig)
    return {'seed':0,'selected_epoch':best_epoch,'epochs_executed':len(rows),'validation_accuracy':accuracy,
        'validation_loss':loss,'optimizer':'fixed_step_LBFGS','lr':.05,'l2_weight':l2,'max_epochs':2000,'patience':100,
        'test_accuracy':None,'test_evaluations':0}, predictions, best_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--embedding-cache', type=Path, required=True)
    parser.add_argument('--l2', type=float, choices=[0.,.0001], required=True)
    parser.add_argument('--resume-head', type=Path)
    args = parser.parse_args()
    verify_originals();seed_everything(0);torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('Actual CUDA required')
    g=dgl.graph(([0],[1]),num_nodes=2).to('cuda');g.ndata['x']=torch.ones(2,1,device='cuda')
    g.update_all(dgl.function.copy_u('x','m'),dgl.function.sum('m','y'))
    if g.ndata['y'][1].item()!=1:raise RuntimeError('Actual DGL CUDA failed')
    del g
    state=read_json_file(args.embedding_cache/'inference_state.json');model_hash=fingerprint(args.checkpoint)
    audit=copy.deepcopy(state['identity']['data']);file=args.embedding_cache/'fused_embeddings.npy'
    if (not state['complete'] or state['identity']['checkpoint_sha256']!=model_hash or audit['dataset']!='WN18RR'
        or audit.get('diagnostic_topology',{}).get('policy')!='train_edges_plus_reverse_preserve_parallel_edges'
        or fingerprint(file)!=state['output_sha256']):raise ValueError('Use the verified complete previously selected WN graph cache')
    manifest=read_json_file(prepared_directory('WN18RR')/'manifest.json')
    for name,digest in manifest['files'].items():
        if fingerprint(prepared_directory('WN18RR')/name)!=digest:raise ValueError('Official data changed')
    embeddings=np.load(file,mmap_mode='r')
    if embeddings.shape!=(manifest['nodes'],1024):raise ValueError('Full-width full-node representation required')
    train_raw=kg_input('WN18RR',embeddings,'train');valid_raw=kg_input('WN18RR',embeddings,'valid')
    mean,scale=training_pair_statistics(embeddings,train_raw.pairs)
    stats_hash=hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);config=copy.deepcopy(saved['config']);del saved
    config.update(name='wn_convex_l2_'+str(args.l2),seed=0,storage=read_json_file(ROOT/'configs/storage.json'))
    config['evaluation']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['evaluation'].update(dataset='WN18RR',task='relation',graph_policy='train_reverse',feature_transform='train_pair_affine',
        optimizer='fixed_step_LBFGS',lr=.05,max_epochs=2000,patience=100,affine_statistics_sha256=stats_hash,
        checkpoint_every=100,selection_split='official_validation',test_used_for_selection=False)
    config['deviations']=list(config.get('deviations',[]))+[
        'Fixed-step LBFGS and train-derived invertible affine coordinates are declared numerical optimization candidates, not confirmed author optimizer. Original concatenation/linear architecture and lr=.05/2000 retained.',
        'Candidate L2 weight is explicitly recorded; nonzero L2 is additional regularization not claimed paper-specified. Test never evaluated in this module.']
    audit['feature_affine']={'statistics_split':'official_train_pairs_only','statistics_sha256':stats_hash,'invertible':True}
    experiment=Experiment(config,audit,args.resume_head or args.checkpoint)
    try:
        train=AffinePairInput(embeddings,train_raw.labels,pairs=train_raw.pairs).set_affine(mean,scale)
        valid=AffinePairInput(embeddings,valid_raw.labels,pairs=valid_raw.pairs).set_affine(mean,scale);valid.cache_validation(400,experiment.logger)
        summary,scaled_prediction,best=train_candidate(train,valid,manifest['relations'],args.l2,experiment,args.resume_head)
        weight,bias=fold_affine_linear(best['weight'].numpy(),best['bias'].numpy(),mean,scale)
        head=torch.nn.Linear(2048,manifest['relations'],dtype=torch.float64).cuda()
        head.load_state_dict({'weight':torch.from_numpy(weight),'bias':torch.from_numpy(bias)})
        raw=DoublePairInput(embeddings,valid_raw.labels,pairs=valid_raw.pairs)
        accuracy,loss,prediction=score_probe(head,raw,save_predictions=True)
        differences=int(np.sum(prediction!=scaled_prediction))
        if differences>2:raise ValueError('Raw linear fold numerical equivalence failed')
        summary.update(scaled_validation_accuracy=summary['validation_accuracy'],validation_accuracy=accuracy,validation_loss=loss,
            folded_validation_disagreements=differences,dataset='WN18RR',task='relation',representation='came',graph_policy='train_reverse',
            encoder_sha256=model_hash,feature_transform='train_pair_affine',affine_statistics_sha256=stats_hash,
            edge_representation='head_tail_concatenation',folded_classifier_evaluation_precision='float64_cuda',
            diagnostic_only=True,paper_reproduction_complete=False)
        atomic_checkpoint(experiment.checkpoint_directory/'probe_best_raw_linear.pt',{'classifier':head.state_dict(),'selected_epoch':summary['selected_epoch'],'config':config,'data_audit':audit})
        np.savez_compressed(experiment.directory/'validation_predictions.npz',prediction=prediction,labels=valid_raw.labels,pairs=valid_raw.pairs)
        np.savez_compressed(experiment.directory/'train_affine_statistics.npz',mean=mean,scale=scale)
        write_json(experiment.directory/'summary.json',summary);experiment.status('completed_convex_kg_validation')
    except BaseException as error:
        experiment.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
    finally:experiment.close()


if __name__=='__main__':main()
