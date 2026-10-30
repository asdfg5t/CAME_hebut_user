"""Validate KG training-edge direction; Appendix C's concatenation linear head stays fixed."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path

import dgl
import numpy as np
import torch

from .author_loader import ROOT, fingerprint, verify_originals
from .atomic_io import read_json_file
from .downstream import ProbeInput, train_linear_probe, score_probe
from .experiment import Experiment, write_json, seed_everything
from .prepared_data import load_prepared, prepared_directory
from .quality_probe import CachedProbeInput, owned_cache_path, remove_owned_cache
from .stream_inference import extract_full_embeddings
from .train import make_model
from .pause_control import TrainingPaused


def training_reverse_graph(graph):
    """OFA's reverse view, preserving parallel train edges and entity IDs."""
    return dgl.add_reverse_edges(graph, copy_ndata=False)


def choose_graph_policy(reference_validation, candidate_validation, min_gain_pp=.5):
    if not all(math.isfinite(v) for v in (reference_validation, candidate_validation, min_gain_pp)):
        raise ValueError('Finite validation values required')
    return 'train_reverse' if 100*(candidate_validation-reference_validation) >= min_gain_pp else 'original'


def validate_confirmation_lock(lock, dataset, model_hash, graph_policy):
    if (lock.get('dataset') != dataset or lock.get('encoder_sha256') != model_hash
            or lock.get('policy') != graph_policy or not lock.get('locked_before_test')
            or lock.get('test_used_for_selection') is not False
            or choose_graph_policy(lock['reference_validation_accuracy'], lock['candidate_validation_accuracy']) != graph_policy
            or lock.get('edge_representation') != 'head_tail_concatenation'
            or lock.get('classifier') != 'linear_cross_entropy_lr_0.05_max2000_patience100'):
        raise ValueError('Invalid validation-only KG policy lock')


def kg_input(dataset, embeddings, split):
    triples = np.load(prepared_directory(dataset)/(split+'_triples.npy'), mmap_mode='r')
    return ProbeInput(embeddings, np.asarray(triples[:,1]), pairs=np.asarray(triples[:,[0,2]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['WN18RR','FB15K237'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--graph-policy', choices=['original','train_reverse'], default='train_reverse')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inference-batch-size', type=int, default=32)
    parser.add_argument('--resume-head', type=Path)
    parser.add_argument('--confirm-test', action='store_true')
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--discard-derived-cache', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required')
    verify_originals(); seed_everything(args.seed); torch.set_num_threads(4)
    model_hash = fingerprint(args.checkpoint)
    if args.confirm_test:
        if args.selection_lock is None:
            raise ValueError('Freeze the validation-only graph policy before test confirmation')
        validate_confirmation_lock(read_json_file(args.selection_lock),args.dataset,model_hash,args.graph_policy)
    check = dgl.graph(([0],[1]),num_nodes=2).to('cuda')
    check.ndata['x'] = torch.ones(2,1,device='cuda')
    check.update_all(dgl.function.copy_u('x','m'),dgl.function.sum('m','y'))
    if check.ndata['y'][1].item() != 1:
        raise RuntimeError('DGL CUDA operation failed')
    del check
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['config']['datasets'] != ['ogbn-products','books-lp','cloth-copurchase']:
        raise ValueError('KG transfer uses the recorded standard pretraining corpus')
    data = load_prepared(args.dataset,require_verified=True,build_graph=True)
    data.audit = copy.deepcopy(data.audit)
    directory = prepared_directory(args.dataset)
    train_triples = np.load(directory/'train_triples.npy',mmap_mode='r')
    original_edges = torch.stack(data.graph.edges()).numpy()
    np.testing.assert_array_equal(original_edges,np.asarray(train_triples[:,[0,2]]).T)
    if args.graph_policy == 'train_reverse':
        before = data.graph.num_edges()
        data.graph = training_reverse_graph(data.graph)
        data.audit['diagnostic_topology'] = {'policy':'train_edges_plus_reverse_preserve_parallel_edges',
            'original_edges':before,'candidate_edges':data.graph.num_edges(),'entity_ids_preserved':True,
            'no_validation_test_edges_added_as_inputs':True,'author_came_loader_confirmed':False,
            'source':'OFA 9045ccdf task_constructor.py:171/247; ofa_datasets.py:436; local pinned archive'}
    storage = read_json_file(ROOT/'configs/storage.json')
    config = copy.deepcopy(saved['config'])
    config.update(name='kg_graph_'+args.dataset,seed=args.seed,storage=storage)
    config['evaluation'] = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['evaluation'].update(task='relation',selection_split='official_validation',
        edge_representation='head_tail_concatenation',test_policy='single_confirmation_after_validation_lock' if args.confirm_test else 'held_out_not_evaluated')
    config['deviations'] = list(config.get('deviations',[]))+[
        'KG graph direction is a validation candidate supported by OFA, not confirmed author CAME loader; Appendix C concatenation linear head retained.']
    experiment = Experiment(config,data.audit,args.resume_head or args.checkpoint)
    cache = owned_cache_path(storage,model_hash,args.dataset,args.graph_policy)
    try:
        model = make_model(config); model.load_state_dict(saved['model'])
        del saved; gc.collect()
        embeddings = extract_full_embeddings(model,data,cache,model_hash,args.inference_batch_size,experiment.logger)
        del model; torch.cuda.empty_cache()
        train = kg_input(args.dataset,embeddings,'train')
        valid = kg_input(args.dataset,embeddings,'valid')
        valid = CachedProbeInput(embeddings,valid.labels,pairs=valid.pairs)
        valid.cache_validation(400,experiment.logger)
        test = kg_input(args.dataset,embeddings,'test') if args.confirm_test else None
        classes = read_json_file(directory/'manifest.json')['relations']
        summary = train_linear_probe(train,valid,test,classes,2000,100,args.seed,experiment,
                    resume=args.resume_head,evaluate_test=args.confirm_test)
        if not args.confirm_test:
            best = torch.load(experiment.checkpoint_directory/'probe_best.pt',map_location='cpu',weights_only=False)
            head = torch.nn.Linear(train.width,classes).cuda();head.load_state_dict(best['classifier'])
            accuracy,_,prediction = score_probe(head,valid,save_predictions=True)
            if accuracy != summary['validation_accuracy']:
                raise ValueError('Saved validation predictions disagree with selected model')
            np.savez_compressed(experiment.directory/'validation_predictions.npz',prediction=prediction,labels=valid.labels,pairs=valid.pairs)
        summary.update(dataset=args.dataset,task='relation',representation='came',graph_policy=args.graph_policy,
            encoder_sha256=model_hash,edge_representation='head_tail_concatenation',diagnostic_only=True,paper_reproduction_complete=False)
        write_json(experiment.directory/'summary.json',summary)
        experiment.status('completed_kg_graph_probe')
        if args.discard_derived_cache:
            embeddings._mmap.close(); del embeddings,train,valid,test;gc.collect()
            remove_owned_cache(cache,storage)
    except BaseException as error:
        experiment.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error))
        raise
    finally:
        experiment.close()


if __name__ == '__main__':
    main()
