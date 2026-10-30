"""Validation-only train-derived optimizer conditioning with a folded raw linear KG head."""
import argparse
import copy
import hashlib
from pathlib import Path
import dgl
import numpy as np
import torch

from .author_loader import ROOT,fingerprint,verify_originals
from .atomic_io import read_json_file
from .downstream import ProbeInput,score_probe,train_linear_probe
from .experiment import Experiment,write_json,seed_everything,atomic_checkpoint
from .feature_affine import training_pair_statistics,fold_affine_linear
from .kg_graph_probe import kg_input
from .prepared_data import prepared_directory
from .pause_control import TrainingPaused
from .quality_probe import CachedProbeInput


class AffinePairInput(CachedProbeInput):
    def set_affine(self,mean,scale):
        self.mean = torch.from_numpy(mean).cuda()
        self.scale = torch.from_numpy(scale).cuda()
        return self

    def batch(self,index):
        features,labels = super().batch(index)
        # Both NumPy-copy transfer and advanced CUDA indexing return private rows.
        # Reuse that storage instead of three simultaneously live full-batch arrays.
        return features.sub_(self.mean).div_(self.scale),labels


class DoublePairInput(ProbeInput):
    def batch(self,index):
        features,labels = super().batch(index)
        return features.double(),labels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['WN18RR','FB15K237'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--embedding-cache',type=Path,required=True)
    parser.add_argument('--graph-policy',choices=['original','train_reverse'],required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--resume-head',type=Path)
    parser.add_argument('--confirm-test',action='store_true')
    parser.add_argument('--selection-lock',type=Path)
    args=parser.parse_args()
    verify_originals();seed_everything(args.seed);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
    check=dgl.graph(([0],[1]),num_nodes=2).to('cuda');check.ndata['x']=torch.ones(2,1,device='cuda')
    check.update_all(dgl.function.copy_u('x','m'),dgl.function.sum('m','y'))
    if check.ndata['y'][1].item()!=1:raise RuntimeError('DGL CUDA failed')
    del check
    model_hash=fingerprint(args.checkpoint)
    state=read_json_file(args.embedding_cache/'inference_state.json')
    cache_file=args.embedding_cache/'fused_embeddings.npy'
    if not state['complete'] or state['identity']['checkpoint_sha256']!=model_hash or fingerprint(cache_file)!=state['output_sha256']:
        raise ValueError('Use the verified complete frozen encoder cache')
    data=copy.deepcopy(state['identity']['data'])
    if data['dataset']!=args.dataset:raise ValueError('Wrong dataset cache')
    cached_policy='train_reverse' if data.get('diagnostic_topology',{}).get('policy')=='train_edges_plus_reverse_preserve_parallel_edges' else 'original'
    if cached_policy!=args.graph_policy:raise ValueError('Wrong KG graph policy cache')
    directory=prepared_directory(args.dataset);manifest=read_json_file(directory/'manifest.json')
    for name,digest in manifest['files'].items():
        if fingerprint(directory/name)!=digest:raise ValueError('Official prepared data changed')
    embeddings=np.load(cache_file,mmap_mode='r')
    if embeddings.shape!=(manifest['nodes'],1024):raise ValueError('Full encoder width and nodes required')
    train_raw=kg_input(args.dataset,embeddings,'train');valid_raw=kg_input(args.dataset,embeddings,'valid')
    mean,scale=training_pair_statistics(embeddings,train_raw.pairs)
    stats_hash=hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest()
    if args.confirm_test:
        if args.selection_lock is None:raise ValueError('Freeze validation policy before confirmation')
        lock=read_json_file(args.selection_lock)
        if (lock.get('dataset')!=args.dataset or lock.get('encoder_sha256')!=model_hash or lock.get('graph_policy')!=args.graph_policy
                or lock.get('affine_statistics_sha256')!=stats_hash or lock.get('feature_transform')!='train_pair_affine'
                or not lock.get('locked_before_test') or lock.get('test_used_for_selection') is not False):
            raise ValueError('Invalid validation-only affine confirmation lock')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=copy.deepcopy(saved['config']);del saved
    config.update(name='kg_affine_'+args.dataset,seed=args.seed,storage=read_json_file(ROOT/'configs/storage.json'))
    config['evaluation']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['evaluation'].update(task='relation',selection_split='official_validation',edge_representation='head_tail_concatenation',
        feature_transform='train_pair_affine',affine_statistics_sha256=stats_hash,folded_classifier_evaluation_precision='float64_cuda')
    config['deviations']=list(config.get('deviations',[]))+[
        'Training-pair-only affine optimizer conditioning is an explicit engineering adaptation. Folded raw classifier is still linear on head/tail concatenation; encoder/head width/LR/loss retained.',
        'Frozen node embeddings and training remain FP32; folded linear evaluation uses CUDA float64 to limit cancellation. Author preprocessing/optimizer details remain unconfirmed.']
    data['feature_affine']={'statistics_split':'official_train_pairs_only','statistics_sha256':stats_hash,'invertible':True,'extra_nonlinear_head':False}
    experiment=Experiment(config,data,args.resume_head or args.checkpoint)
    try:
        np.savez_compressed(experiment.directory/'train_affine_statistics.npz',mean=mean,scale=scale)
        train=AffinePairInput(embeddings,train_raw.labels,pairs=train_raw.pairs).set_affine(mean,scale)
        valid=AffinePairInput(embeddings,valid_raw.labels,pairs=valid_raw.pairs).set_affine(mean,scale)
        valid.cache_validation(400,experiment.logger)
        classes=manifest['relations']
        summary=train_linear_probe(train,valid,None,classes,2000,100,args.seed,experiment,resume=args.resume_head,evaluate_test=False)
        best=torch.load(experiment.checkpoint_directory/'probe_best.pt',map_location='cpu',weights_only=False)
        scaled=torch.nn.Linear(train.width,classes).cuda();scaled.load_state_dict(best['classifier'])
        _,_,scaled_predictions=score_probe(scaled,valid,save_predictions=True)
        weight,bias=fold_affine_linear(best['classifier']['weight'].numpy(),best['classifier']['bias'].numpy(),mean,scale)
        raw=torch.nn.Linear(train.width,classes,dtype=torch.float64).cuda()
        raw.load_state_dict({'weight':torch.from_numpy(weight),'bias':torch.from_numpy(bias)})
        raw_valid=DoublePairInput(embeddings,valid_raw.labels,pairs=valid_raw.pairs)
        accuracy,loss,prediction=score_probe(raw,raw_valid,save_predictions=True)
        disagreements=int(np.sum(prediction!=scaled_predictions))
        if disagreements>2:raise ValueError('Folded/raw affine numerical equivalence failed; inspect before testing')
        summary['scaled_validation_accuracy']=summary['validation_accuracy']
        summary.update(validation_accuracy=accuracy,validation_loss=loss,folded_validation_disagreements=disagreements,
            dataset=args.dataset,task='relation',representation='came',graph_policy=args.graph_policy,seed=args.seed,encoder_sha256=model_hash,
            feature_transform='train_pair_affine',affine_statistics_sha256=stats_hash,edge_representation='head_tail_concatenation',
            folded_classifier_evaluation_precision='float64_cuda',diagnostic_only=True,paper_reproduction_complete=False)
        np.savez_compressed(experiment.directory/'validation_predictions.npz',prediction=prediction,labels=valid_raw.labels,pairs=valid_raw.pairs)
        atomic_checkpoint(experiment.checkpoint_directory/'probe_best_raw_linear.pt',{'classifier':raw.state_dict(),'selected_epoch':summary['selected_epoch'],
            'affine_statistics_sha256':stats_hash,'folded_to_raw_head_tail_concatenation':True,'config':config,'data_audit':data})
        if args.confirm_test:
            test=kg_input(args.dataset,embeddings,'test');raw_test=DoublePairInput(embeddings,test.labels,pairs=test.pairs)
            summary['test_accuracy'],summary['test_loss'],prediction=score_probe(raw,raw_test,save_predictions=True)
            summary['test_evaluations']=1
            np.savez_compressed(experiment.directory/'test_predictions.npz',prediction=prediction,labels=test.labels,pairs=test.pairs)
        write_json(experiment.directory/'summary.json',summary);experiment.status('completed_kg_affine_probe')
    except BaseException as error:
        experiment.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
    finally:experiment.close()


if __name__=='__main__':main()
