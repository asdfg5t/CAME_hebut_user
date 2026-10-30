"""Finite new diagnostics; author sources and all historical runs stay immutable."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
PROFILE = ROOT/'configs/all_unresolved_20261018.json'
OUT = ROOT/'artifacts/all_unresolved_20261018'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def storage(): return read(ROOT/'configs/storage.json')


def encoder():
    from came_repro.author_loader import fingerprint
    path = Path(read(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    if fingerprint(path) != read(PROFILE)['encoder_sha256']: raise ValueError('Protected encoder changed')
    return path


def compact_directory():
    return Path(storage()['embedding_cache_root'])/'structural_indexes/products_csc32_v1'


def wn_directory():
    return Path(storage()['embedding_cache_root'])/'unresolved_20261018/WN18RR_unit_l2'


def acceptance():
    import numpy as np
    import torch
    from came_repro.protocol_repair import check_cuda
    from came_repro.experiment import Experiment, seed_everything
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro.cloth_regularized_train import train_regularized
    from came_repro.pause_control import TrainingPaused
    import came_repro.pause_control as pauses
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    from cloth_wn_autonomous_backend import assert_same
    from came_repro.compact_csc import build_index, load_graph
    from came_repro.prepared_data import PreparedData
    from came_repro.stream_inference import batch_full_neighbor_embeddings
    from came_repro.train import make_model
    import dgl
    check_cuda(); torch.set_num_threads(4); seed_everything(713)
    recipe = copy.deepcopy(read(PROFILE)['cloth']['recipe']); recipe.update(batch_size=3, max_epochs=3)
    gpu = torch.randn(16, 1024, device='cuda')
    train = dict(source_node=np.arange(6), target_node=np.arange(6)+1)
    valid = dict(source_node=np.array([8, 9]), target_node=np.array([9, 10]),
                 target_node_neg=np.tile(np.arange(16), (2, 10))[:, :150])
    config = dict(name='regularized_resume_fixture', seed=17, storage=storage(),
                  evaluation=dict(recipe=recipe, fixture_only=True))
    audit = dict(fixture_only_not_benchmark=True)
    reference = Experiment(config, audit)
    try:
        with use_cached_vectors(): train_regularized(gpu, train, valid, recipe, 17, reference)
        reference.status('completed_fixture_reference')
    finally: reference.close()
    original = pauses.pause_requested; calls = [0]
    def pause_two(): calls[0] += 1; return calls[0] >= 2
    pauses.pause_requested = pause_two; paused = Experiment(config, audit)
    try:
        try:
            with use_cached_vectors(): train_regularized(gpu, train, valid, recipe, 17, paused)
        except TrainingPaused: paused.status('paused_fixture')
        else: raise ValueError('Pause fixture did not stop')
    finally: paused.close(); pauses.pause_requested = original
    cp = paused.checkpoint_directory/'link_head_last.pt'; resumed = Experiment(config, audit, cp)
    try:
        with use_cached_vectors(): train_regularized(gpu, train, valid, recipe, 17, resumed, cp)
        resumed.status('completed_fixture_resume')
    finally: resumed.close()
    a = torch.load(str(reference.checkpoint_directory/'link_head_last.pt'), map_location='cpu', weights_only=False)
    b = torch.load(str(resumed.checkpoint_directory/'link_head_last.pt'), map_location='cpu', weights_only=False)
    for key in ('head', 'optimizer', 'rng', 'sampling_generator', 'history_rows', 'best_state', 'selected_epoch', 'next_epoch'):
        assert_same(a[key], b[key])
    # Actual full original-model evaluation versus exact compact inbound blocks.
    saved = torch.load(str(encoder()), map_location='cpu', weights_only=False, mmap=True)
    model = make_model(saved['config']); model.load_state_dict(saved['model']); model.eval()
    edges = np.array([[0, 1], [0, 1], [1, 2], [2, 0], [3, 3], [3, 3], [4, 1]])
    directory = OUT/'csc_fixture'; build_index(edges, 7, directory, 'fixture')
    graph = load_graph(directory)
    features = np.random.default_rng(19).normal(size=(7, 1024)).astype(np.float32)
    raw = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=7)
    with torch.no_grad():
        expected = model(raw.to('cuda'), torch.from_numpy(features[:, :512]).cuda(),
                         torch.from_numpy(features[:, 512:]).cuda(), getz3=True, compute_loss=False).cpu().numpy()
        data = PreparedData(graph, features, None, {}, {})
        actual = []
        for start in range(0, 7, 3):
            inputs, targets, blocks = dgl.dataloading.MultiLayerFullNeighborSampler(2).sample_blocks(
                graph, torch.arange(start, min(7, start+3), dtype=torch.int32))
            actual.append(batch_full_neighbor_embeddings(model, data, blocks, inputs).cpu().numpy())
        actual = np.concatenate(actual)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all(): raise FloatingPointError('Nonfinite compact fixture')
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    from came_repro.regularized_link_head import RegularizedLinkHead
    from came_repro.lp_validation_repair import LayerNormLinkHead
    x = torch.randn(8, 1024, device='cuda'); h = RegularizedLinkHead().cuda().eval(); old = LayerNormLinkHead().cuda().eval()
    old.load_state_dict(h.state_dict())
    with torch.no_grad(): torch.testing.assert_close(h(x, x), old(x, x), rtol=0, atol=0)
    unit = torch.nn.functional.normalize(x, dim=1)
    torch.testing.assert_close(unit.norm(dim=1), torch.ones(8, device='cuda'))
    atomic_write_json(OUT/'acceptance.json', dict(passed=True, actual_Torch_DGL_CUDA=True,
        dropout_full_Adam_RNG_sampler_resume_bitexact=True, dropout_eval_head_identical=True,
        compact_full_author_model_max_abs_difference=float(np.max(np.abs(actual-expected))),
        isolated_parallel_edges_and_multiple_selfloops_verified=True, normalization_CUDA_unit_norm_verified=True,
        fixture_only_not_benchmark=True, fixture_runs=[str(e.directory) for e in (reference, paused, resumed)],
        profile_sha256=fingerprint(PROFILE), backend_sha256=fingerprint(__file__),
        new_source_sha256={str(ROOT/'src/came_repro'/n): fingerprint(ROOT/'src/came_repro'/n)
                          for n in ('compact_csc.py', 'regularized_link_head.py', 'cloth_regularized_train.py')}))


def cloth(resume):
    import torch
    from came_repro.author_loader import fingerprint
    from came_repro.lp_validation_repair import load_verified_cache
    from came_repro.cloth_regularized_train import train_regularized
    from came_repro.experiment import Experiment
    from came_repro.atomic_io import atomic_write_json
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro.pause_control import TrainingPaused
    p = read(PROFILE); reference = ROOT/p['cloth']['reference_run']; s = read(reference/'summary.json')
    cache = Path(s['embedding_cache']); gpu, audit = load_verified_cache('cloth-copurchase', cache, p['encoder_sha256'])
    if audit['diagnostic_topology']['policy'] != 'upstream_train_bidirected_unique': raise ValueError('Graph changed')
    raw = ROOT/'data/raw/mm_graph/cloth-copurchase/lp-edge-split.pt'
    if fingerprint(raw) != audit['raw_sha256']['lp-edge-split.pt']: raise ValueError('Raw split changed')
    split = torch.load(str(raw), map_location='cpu', weights_only=False, mmap=True)
    config = read(reference/'config.json'); config.update(name='cloth_dropout_candidate', seed=0)
    recipe = p['cloth']['recipe']; config['evaluation'].update(recipe, all_unresolved_profile_sha256=fingerprint(PROFILE))
    config['deviations'] = list(config.get('deviations', []))+['Training-only dropout0.3; no new evaluation parameters; CAME LP dropout is unreleased.']
    e = Experiment(config, audit, resume or encoder())
    try:
        with use_cached_vectors(): result = train_regularized(gpu, split['train'], split['valid'], recipe, 0, e, resume)
        result.update(dataset='cloth-copurchase', task='link', representation='came', encoder_sha256=p['encoder_sha256'],
            embedding_cache=str(cache), graph_policy='upstream_train_bidirected_unique', diagnostic_only=True,
            paper_reproduction_complete=False, new_training_dropout=.3)
        atomic_write_json(e.directory/'summary.json', result); e.status('completed_regularized_validation_only')
    except BaseException as error:
        e.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error)); raise
    finally: e.close()


def wn(resume):
    import numpy as np
    import torch
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro import kg_affine_probe as kg
    source = Path(storage()['embedding_cache_root'])/'protocol_repair'/read(PROFILE)['encoder_sha256'][:20]/read(PROFILE)['wn']['source_cache_suffix']
    old = read(source/'inference_state.json'); original = source/'fused_embeddings.npy'
    if (not old['complete'] or old['identity']['checkpoint_sha256'] != read(PROFILE)['encoder_sha256']
            or old['identity']['data']['dataset'] != 'WN18RR' or fingerprint(original) != old['output_sha256']):
        raise ValueError('Original WN cache changed')
    target = wn_directory(); target.mkdir(parents=True, exist_ok=True); state_path = target/'inference_state.json'
    state = copy.deepcopy(old)
    state['identity']['data']['representation_transform'] = dict(method='unit_L2_per_frozen_node_FP32_CUDA', source_sha256=old['output_sha256'], labels_used=False)
    state['identity']['representation_postprocess'] = dict(method='unit_L2', epsilon=1e-12, source_output_sha256=old['output_sha256'])
    state.pop('prefix_sha256', None)
    state['derived_from_state_sha256'] = fingerprint(source/'inference_state.json')
    if state_path.exists():
        existing = read(state_path)
        if existing['identity'] != state['identity'] or fingerprint(target/'fused_embeddings.npy') != existing['output_sha256']:
            raise ValueError('Normalized WN cache changed')
    else:
        from came_repro.storage_budget import guard_allocation
        matrix = np.load(original, mmap_mode='r'); guard_allocation(target/'fused_embeddings.npy', matrix.nbytes+2**20)
        output = np.lib.format.open_memmap(target/'fused_embeddings.npy', mode='w+', dtype=np.float32, shape=matrix.shape)
        for start in range(0, len(matrix), 4096):
            batch = torch.from_numpy(np.array(matrix[start:start+4096], copy=True)).cuda()
            if not torch.isfinite(batch).all(): raise FloatingPointError('Nonfinite WN representation')
            output[start:start+len(batch)] = torch.nn.functional.normalize(batch, dim=1).cpu().numpy()
        output.flush(); output._mmap.close(); state['output_sha256'] = fingerprint(target/'fused_embeddings.npy')
        atomic_write_json(state_path, state)
    original_factory = kg.Experiment
    def factory(config, audit, parent):
        config = copy.deepcopy(config); config['name'] = 'wn_unit_l2_candidate'
        config['evaluation'].update(representation_transform='unit_L2_per_node', new_profile_sha256=fingerprint(PROFILE))
        config['deviations'] = list(config.get('deviations', []))+['Frozen node L2 normalization is an engineering preprocessing candidate, not author-confirmed.']
        return original_factory(config, audit, parent)
    kg.Experiment = factory; previous = sys.argv
    sys.argv = ['kg_affine_probe', '--dataset', 'WN18RR', '--checkpoint', str(encoder()),
                '--embedding-cache', str(target), '--graph-policy', 'train_reverse', '--seed', '0']
    if resume: sys.argv += ['--resume-head', str(resume)]
    try: kg.main()
    finally: kg.Experiment = original_factory; sys.argv = previous


def product_data():
    import numpy as np
    from came_repro.prepared_data import load_prepared, prepared_directory
    from came_repro.compact_csc import build_index, load_graph
    from came_repro.author_loader import fingerprint
    data = load_prepared('ogbn-products', build_graph=False)
    file = prepared_directory('ogbn-products')/'edge_index.npy'
    manifest = read(prepared_directory('ogbn-products')/'manifest.json')
    index = build_index(np.load(file, mmap_mode='r'), data.audit['nodes'], compact_directory(), manifest['files']['edge_index.npy'])
    if fingerprint(file) != manifest['files']['edge_index.npy']: raise ValueError('Raw topology changed')
    data.graph = load_graph(compact_directory())
    return data, index


def products_calibration():
    import gc
    import time
    import numpy as np
    import torch
    import dgl
    import psutil
    from came_repro.train import make_model
    from came_repro.stream_inference import batch_full_neighbor_embeddings
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.experiment import Experiment
    p = read(PROFILE); saved = torch.load(str(encoder()), map_location='cpu', weights_only=False, mmap=True)
    config = copy.deepcopy(saved['config']); config.update(name='products_compact_capacity', seed=0, storage=storage())
    config['evaluation'] = dict(no_test_scoring=True, structural_storage='int32_CSC', profile_sha256=fingerprint(PROFILE))
    data, index = product_data(); e = Experiment(config, data.audit, encoder())
    try:
        model = make_model(config); model.load_state_dict(saved['model']); model.eval(); del saved
        ptr = np.load(compact_directory()/'indptr.npy', mmap_mode='r'); degrees = np.diff(ptr)
        top = np.argsort(degrees)[-4:][::-1]; random = np.random.default_rng(217).choice(data.audit['nodes'], 12, replace=False)
        prefix_cache = Path(storage()['embedding_cache_root'])/p['encoder_sha256'][:20]/'ogbn-products'
        state = read(prefix_cache/'inference_state.json'); cursor = state['cursor']
        if (state['identity']['checkpoint_sha256'] != p['encoder_sha256']
                or state['identity']['data'] != data.audit or cursor < 16):
            raise ValueError('Original Products prefix identity changed or missing')
        prefix_ids = np.linspace(0, cursor-1, 16, dtype=np.int64)
        probes = np.unique(np.concatenate((top, random, prefix_ids)))
        def score(graph, ids, batch):
            rows = []; sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            with torch.no_grad():
                for start in range(0, len(ids), batch):
                    inputs, targets, blocks = sampler.sample_blocks(graph, torch.tensor(ids[start:start+batch], dtype=graph.idtype))
                    value = batch_full_neighbor_embeddings(model, data, blocks, inputs)
                    if not torch.isfinite(value).all(): raise FloatingPointError('Nonfinite Products calibration')
                    rows.append(value.cpu().numpy())
            return np.concatenate(rows)
        legacy = data.graph.long(); expected = score(legacy, probes, 1); del legacy; gc.collect()
        results = []; selected = None
        for batch in p['products']['probe_batch_sizes']:
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
                actual = score(data.graph, probes, batch)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
                results.append(dict(batch=batch, passed=True, seconds=time.perf_counter()-start,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(), maximum_difference=float(np.max(np.abs(actual-expected)))))
                e.metric(dict(step=batch, elapsed_seconds=results[-1]['seconds'],
                    peak_cuda_mib=results[-1]['peak_cuda_bytes']/2**20, max_abs_difference=results[-1]['maximum_difference']))
                selected = batch
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache(); break
        if selected is None: raise RuntimeError('Compact exact Products inference cannot fit GPU')
        old = np.load(prefix_cache/'fused_embeddings.npy', mmap_mode='r')
        prefix_actual = score(data.graph, prefix_ids, min(selected, 4))
        np.testing.assert_allclose(prefix_actual, old[prefix_ids], rtol=2e-5, atol=2e-5)
        start = time.perf_counter(); sample_start = min(cursor, data.audit['nodes']-p['products']['long_sample_rows'])
        sample = np.arange(sample_start, sample_start+p['products']['long_sample_rows'])
        score(data.graph, sample, selected); seconds = time.perf_counter()-start
        process = psutil.Process(); memory = process.memory_info(); private = getattr(memory, 'private', memory.rss)
        throughput = len(sample)/seconds; total = psutil.virtual_memory().total
        e.metric(dict(step=32, elapsed_seconds=seconds, peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
            max_abs_difference=float(np.max(np.abs(prefix_actual-old[prefix_ids])))))
        ready = private < p['products']['maximum_process_memory_fraction']*total and throughput >= p['products']['minimum_measured_nodes_per_second_for_full_start']
        summary = dict(passed=True, ready_for_full_start=ready, selected_batch_size=selected, measurements=results,
            measured_contiguous_rows=len(sample), nodes_per_second=throughput, private_bytes=private, rss_bytes=memory.rss,
            full_graph_nodes=data.audit['nodes'], full_graph_edges_with_selfloops=index['edges'], width=1024,
            prefix_cursor=cursor, original_prefix_sample_verified=True, full_official_test_nodes=2213091, test_evaluations=0,
            profile_sha256=fingerprint(PROFILE), index_manifest_sha256=fingerprint(compact_directory()/'manifest.json'))
        atomic_write_json(e.directory/'summary.json', summary); atomic_write_json(OUT/'products_calibration.json', summary)
        e.status('completed_compact_calibration'); e.logger.info('Compact Products resources: %s', summary)
    except BaseException as error: e.status('failed', error=str(error)); raise
    finally: e.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['acceptance', 'cloth', 'wn', 'products_calibration', 'products_full'], required=True)
    parser.add_argument('--resume-head', type=Path); args = parser.parse_args()
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.protocol_repair import check_cuda
    import torch
    verify_originals(); check_cuda(); torch.set_num_threads(4); OUT.mkdir(parents=True, exist_ok=True)
    if args.stage != 'acceptance':
        accepted = read(OUT/'acceptance.json')
        if not accepted['passed'] or accepted['backend_sha256'] != fingerprint(__file__) or accepted['profile_sha256'] != fingerprint(PROFILE):
            raise ValueError('Matching actual CUDA and complete-resume acceptance required')
        for path, sha in accepted['new_source_sha256'].items():
            if fingerprint(path) != sha: raise ValueError('Accepted source changed')
    if args.stage == 'acceptance': acceptance()
    elif args.stage == 'cloth': cloth(args.resume_head)
    elif args.stage == 'wn': wn(args.resume_head)
    elif args.stage == 'products_calibration': products_calibration()
    else:
        from unresolved_products_full_20261018 import main as full
        full(args.resume_head)


if __name__ == '__main__': main()
