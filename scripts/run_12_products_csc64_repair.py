"""Bounded exact Products repair: native int64 CSC sampling, full FP32 CUDA.

Reuses a hash-pinned serial controller and the unchanged full inference writer.
No training recipe, graph edge, feature width or official split is changed.
"""
import copy
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    sys.path.insert(0, str(ROOT/'scripts'))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
OUT = ROOT/'artifacts/products_csc64_20261018'
PROFILE = ROOT/'configs/products_csc64_20261018.json'
LEDGER = ROOT/'artifacts/workflow_state/products_csc64_20261018_current.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_dependencies():
    from came_repro.author_loader import fingerprint
    plan = read(PROFILE)
    for path, digest in plan['frozen_dependencies'].items():
        if fingerprint(ROOT/path) != digest:
            raise ValueError('Pinned diagnostic dependency changed: '+path)
    if (plan['full_nodes'] != 2449029 or plan['official_test_nodes'] != 2213091
            or plan['width'] != 1024 or plan['maximum_new_test_evaluations'] != 1):
        raise ValueError('Full official Products protocol required')
    return plan


def native_csc64(graph):
    import torch
    graph = graph.long().formats('csc')
    if graph.idtype != torch.int64 or graph.device.type != 'cpu':
        raise ValueError('Native CPU int64 CSC required')
    return graph


def validate_sampled_blocks(blocks, inputs):
    """Check all local/global mappings BEFORE sending data to CUDA."""
    import torch
    import dgl
    if len(blocks) != 2 or inputs.dtype != torch.int64:
        raise ValueError('Two native int64 full-neighbor blocks required')
    for block in blocks:
        if not block.is_block or block.idtype != torch.int64 or block.device.type != 'cpu':
            raise ValueError('Native CPU int64 blocks required')
        source, target = block.edges(order='eid')
        if (len(source) and (source.min() < 0 or source.max() >= block.num_src_nodes()
                            or target.min() < 0 or target.max() >= block.num_dst_nodes())):
            raise ValueError('Sampled block local edge ID out of range')
        if not torch.equal(block.srcdata[dgl.NID][:block.num_dst_nodes()], block.dstdata[dgl.NID]):
            raise ValueError('Destination nodes must be the source prefix')
    if (not torch.equal(inputs, blocks[0].srcdata[dgl.NID])
            or not torch.equal(blocks[0].dstdata[dgl.NID], blocks[1].srcdata[dgl.NID])):
        raise ValueError('Sampled block feature/node ordering mismatch')


def run_stage(stage, resume=None):
    # Set before creating a CUDA context; diagnostics localize asynchronous errors.
    if stage != 'full':
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    plan = validate_dependencies()
    import torch
    import numpy as np
    import dgl
    from came_repro import experiment as experiments, stream_inference as stream
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.atomic_io import atomic_write_json
    from came_repro.protocol_repair import check_cuda
    import unresolved_backend_20261018 as base
    verify_originals(); check_cuda(); torch.set_num_threads(4); OUT.mkdir(parents=True, exist_ok=True)
    old_data, old_out, old_factory, old_forward = base.product_data, base.OUT, experiments.Experiment, stream.batch_full_neighbor_embeddings
    base.OUT = OUT

    def product_data():
        data, index = old_data()
        data.graph = native_csc64(data.graph)
        gc.collect()
        return data, index

    def factory(config, audit, parent=None):
        config = copy.deepcopy(config)
        config['name'] = 'products_csc64_'+config['name']
        config.setdefault('evaluation', {}).update(structural_storage='disk_int32_CSC_native_CPU_GPU_int64',
            compatibility_profile_sha256=fingerprint(PROFILE), adapter_sha256=fingerprint(__file__),
            CUDA_LAUNCH_BLOCKING=os.environ.get('CUDA_LAUNCH_BLOCKING', 'unset'))
        config['deviations'] = [value for value in config.get('deviations', [])
                                if not value.startswith('Structural int32 CSC storage only;')]+[
            'Compact index stored int32; complete CPU CSC and native sampled CUDA blocks use int64. Full topology, FP32 and width1024 unchanged.',
            'Resource/compatibility reconstruction; author settings unchanged; root cause not yet established.']
        return old_factory(config, audit, parent)

    calls = 0
    def guarded_forward(model, data, blocks, inputs):
        nonlocal calls
        validate_sampled_blocks(blocks, inputs); calls += 1
        cursor = dict(call=calls, targets=blocks[-1].dstdata[dgl.NID].tolist(),
            block_shapes=[dict(src=b.num_src_nodes(), dst=b.num_dst_nodes(), edges=b.num_edges()) for b in blocks],
            status='before_original_CUDA_forward')
        atomic_write_json(OUT/'cuda_probe_cursor.json', cursor)
        value = old_forward(model, data, blocks, inputs); torch.cuda.synchronize()
        atomic_write_json(OUT/'cuda_probe_cursor.json', {**cursor, 'status':'completed_original_CUDA_forward'})
        return value

    base.product_data = product_data; experiments.Experiment = factory
    if stage != 'full': stream.batch_full_neighbor_embeddings = guarded_forward
    try:
        if stage == 'small':
            from came_repro.compact_csc import build_index, load_graph
            from came_repro.prepared_data import PreparedData
            from came_repro.train import make_model
            saved = torch.load(str(base.encoder()), map_location='cpu', weights_only=False, mmap=True)
            model = make_model(saved['config']); model.load_state_dict(saved['model']); model.eval()
            edges = np.array([[0,1],[0,1],[1,2],[2,0],[3,3],[3,3],[4,2]])
            index = OUT/'fixture_csc'; build_index(edges, 7, index, 'native64_fixture')
            graph = native_csc64(load_graph(index))
            features = np.random.default_rng(812).normal(size=(7,1024)).astype(np.float32)
            data = PreparedData(graph, features, None, {}, {})
            with torch.no_grad():
                raw = dgl.graph((edges[:,0], edges[:,1]), num_nodes=7).to('cuda')
                expected = model(raw, torch.from_numpy(features[:,:512]).cuda(), torch.from_numpy(features[:,512:]).cuda(),
                                 getz3=True, compute_loss=False).cpu().numpy()
                inputs, targets, blocks = dgl.dataloading.MultiLayerFullNeighborSampler(2).sample_blocks(graph, torch.arange(7))
                actual = guarded_forward(model, data, blocks, inputs).cpu().numpy()
            if not np.isfinite(actual).all() or not np.isfinite(expected).all(): raise FloatingPointError('Nonfinite native64 fixture')
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            atomic_write_json(OUT/'small_acceptance.json', dict(passed=True, actual_Torch_DGL_CUDA=True,
                CPU_graph='int64_CSC', GPU_blocks='native_int64', feature_precision='fp32',
                max_abs_difference=float(np.max(np.abs(actual-expected))), fixture_not_benchmark=True,
                adapter_sha256=fingerprint(__file__), compatibility_profile_sha256=fingerprint(PROFILE)))
        else:
            accepted = read(OUT/'small_acceptance.json')
            if (not accepted['passed'] or accepted['adapter_sha256'] != fingerprint(__file__)
                    or accepted['compatibility_profile_sha256'] != fingerprint(PROFILE)):
                raise ValueError('Matching actual native64 CUDA acceptance required')
            if stage == 'calibration': base.products_calibration()
            else:
                receipt = read(OUT/'products_calibration.json')
                if receipt['private_bytes'] >= .8*__import__('psutil').virtual_memory().total:
                    raise ValueError('Native64 real calibration exceeds RAM budget')
                from unresolved_products_full_20261018 import main as full
                full(resume)
    finally:
        base.product_data, base.OUT = old_data, old_out
        experiments.Experiment, stream.batch_full_neighbor_embeddings = old_factory, old_forward


def configure_relay():
    """Reuse the frozen one-shot serial controller, not its failed computation."""
    import run_11_products_block64_repair as relay
    relay.__file__ = str(Path(__file__).resolve())
    relay.OUT, relay.PROFILE, relay.LEDGER = OUT, PROFILE, LEDGER
    relay.run_stage = run_stage
    return relay


def main():
    plan = validate_dependencies()
    if not LEDGER.exists() and '--stage' not in sys.argv:
        # The reused controller also deducts the original main queue's activity.
        # Charge the previous failed compatibility queue exactly once, too.
        from came_repro.atomic_io import atomic_write_json
        prior = read(ROOT/'artifacts/workflow_state/all_unresolved_20261018_current.json')
        sources = dict(prior['identity']['sources']); sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        identity = dict(sources=sources, base_profile_sha256=prior['identity']['profile_sha256'],
                        compatibility_profile_sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest())
        atomic_write_json(LEDGER, dict(identity=identity, state='waiting_for_prior_serial_queue', records={},
            active_seconds=plan['previous_failed_compatibility_active_seconds'],
            carried_external_activity_seconds=plan['previous_failed_compatibility_active_seconds']))
    configure_relay().main()


if __name__ == '__main__': main()
