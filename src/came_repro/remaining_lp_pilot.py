"""One complete LP seed with exact FP32 CUDA row caching and official candidates."""
import argparse
import copy
import gc
import time
from contextlib import contextmanager
from pathlib import Path

import dgl
import numpy as np
import torch

from . import link_evaluation as link
from .atomic_io import read_json_file
from .author_loader import ROOT, fingerprint, verify_originals
from .experiment import Experiment, seed_everything, write_json
from .pause_control import TrainingPaused
from .prepared_data import load_prepared
from .stream_inference import extract_full_embeddings
from .train import make_model


def cached_vectors(embeddings, index):
    if not isinstance(embeddings, torch.Tensor) or embeddings.device.type != 'cuda' or embeddings.dtype != torch.float32:
        raise ValueError('The exact row cache must be a CUDA FP32 tensor')
    return embeddings[torch.as_tensor(np.asarray(index, dtype=np.int64), device='cuda')]


@contextmanager
def use_cached_vectors():
    # Infrastructure adapter in this child only; upstream files and the LP head/ranking math are unchanged.
    original = link.vectors
    link.vectors = cached_vectors
    try:
        yield
    finally:
        link.vectors = original


def verify_row_cache_equivalence():
    """Duplicate indices, official-style candidates and tied ranks retain exact values."""
    rng = np.random.default_rng(731)
    matrix = rng.normal(size=(17, 1024)).astype(np.float32)
    cached = torch.from_numpy(matrix).cuda()
    ids = np.array([[1, 2, 1], [16, 0, 3]])
    if not torch.equal(link.vectors(matrix, ids), cached_vectors(cached, ids)):
        raise ValueError('Row caching changes feature values or order')
    head = link.LinkHead(1024).cuda()
    split = {'source_node':np.array([0, 1, 2]), 'target_node':np.array([3, 4, 5]),
             'target_node_neg':np.stack([np.resize(row,150) for row in
                 (np.array([3,6,7]), np.array([4,8,8]), np.array([5,9,10]))])}
    overlap = np.array([False, True, False])
    original = link.evaluate_split(head, matrix, split, 2, overlap)
    with use_cached_vectors():
        candidate = link.evaluate_split(head, cached, split, 2, overlap)
    if original != candidate:
        raise ValueError('Exact row-cache official ranking equivalence failed')
    return {'duplicate_and_shaped_indices_exact':True, 'official_mrr_hits_and_reverse_subset_exact':True,
            'precision':'fp32', 'width':1024, 'fixed_negatives_per_query':150,
            'candidate_queries':3, 'benchmark_only_not_paper_result':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['sports-copurchase', 'cloth-copurchase', 'books-lp'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--resume-head', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--inference-batch-size', type=int, default=32)
    args = parser.parse_args()
    verify_originals(); seed_everything(args.seed); torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA is required')
    graph = dgl.graph(([0], [1]), num_nodes=2).to('cuda')
    graph.ndata['x'] = torch.ones(2, 1, device='cuda')
    graph.update_all(dgl.function.copy_u('x', 'm'), dgl.function.sum('m', 'y'))
    if graph.ndata['y'][1].item() != 1:
        raise RuntimeError('Actual DGL CUDA failed')
    del graph
    equivalence = verify_row_cache_equivalence()
    seed_everything(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = copy.deepcopy(saved['config'])
    if config['datasets'] != ['ogbn-products', 'books-lp', 'cloth-copurchase']:
        raise ValueError('Recorded standard joint-pretraining corpus required')
    model_hash = fingerprint(args.checkpoint)
    data = load_prepared(args.dataset, require_verified=True, build_graph=True)
    raw = ROOT/'data/raw/mm_graph'/args.dataset/'lp-edge-split.pt'
    if fingerprint(raw) != data.audit['raw_sha256']['lp-edge-split.pt']:
        raise ValueError('Official immutable LP split changed')
    split = torch.load(raw, map_location='cpu', weights_only=False)
    source, target = data.graph.edges()
    np.testing.assert_array_equal(source.numpy(), np.asarray(split['train']['source_node']))
    np.testing.assert_array_equal(target.numpy(), np.asarray(split['train']['target_node']))
    del source, target
    storage = read_json_file(ROOT/'configs/storage.json')
    config.update(name='lp_single_'+args.dataset, seed=args.seed, storage=storage)
    config['evaluation'] = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config['evaluation'].update(task='link', representation='came', encoder_sha256=model_hash,
        head='hadamard_2layer_mlp', head_is_reconstructed=True, feature_cache_mode='complete_cuda_fp32_rows',
        train_batch_size=4096, evaluation_batch_size=64, lr=.001, test_used_for_selection=False)
    config['deviations'] = list(config.get('deviations', []))+[
        'LP head, optimizer, LR, negative sampling and patience are explicit reconstruction assumptions; not confirmed author settings.',
        'Entire frozen FP32 row matrix cached on CUDA, with exact input/ranking equivalence checked before the real run; no candidate or graph changes.']
    experiment = Experiment(config, data.audit, args.resume_head or args.checkpoint)
    try:
        write_json(experiment.directory/'cuda_row_cache_equivalence.json', equivalence)
        model = make_model(config); model.load_state_dict(saved['model']); del saved; gc.collect()
        cache = Path(storage['embedding_cache_root'])/model_hash[:20]/args.dataset
        embeddings = extract_full_embeddings(model, data, cache, model_hash, args.inference_batch_size, experiment.logger)
        del model, data.graph; gc.collect(); torch.cuda.empty_cache()
        required = embeddings.size*4
        free, _ = torch.cuda.mem_get_info()
        if required+int(1.5*2**30) > free:
            raise RuntimeError('Full FP32 CUDA cache exceeds explicit 1.5 GiB scoring reserve; review capacity before proceeding')
        gpu = torch.empty(embeddings.shape, dtype=torch.float32, device='cuda')
        for start in range(0, len(embeddings), 8192):
            stop = min(start+8192, len(embeddings))
            gpu[start:stop] = torch.from_numpy(np.array(embeddings[start:stop], dtype=np.float32, copy=True)).cuda()
        experiment.logger.info('Full frozen row cache on CUDA: %.3f GiB; all official candidates retained', required/2**30)
        with use_cached_vectors():
            summary = link.train_link_head(gpu, split, args.epochs, args.patience, args.seed, experiment, resume=args.resume_head)
        summary.update(dataset=args.dataset, task='link', representation='came', seed=args.seed,
            encoder_sha256=model_hash, feature_cache_mode='complete_cuda_fp32_rows', diagnostic_only=True,
            elapsed_wall_seconds=time.perf_counter()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            complete_declared_subset_pretraining=True, paper_reproduction_complete=False)
        write_json(experiment.directory/'summary.json', summary)
        experiment.status('completed_single_seed_lp_pilot')
    except BaseException as error:
        experiment.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error)); raise
    finally:
        experiment.close()


if __name__ == '__main__':
    main()
