"""Tests target fidelity, leakage, routing gradients and checkpoint semantics."""
import json
import sys

import dgl
import numpy as np
import pytest
import torch

from came_repro.author_loader import ROOT, load_author_model, verify_originals
from came_repro.data import GraphData, PPRSampler, load_dataset
from came_repro.experiment import seed_everything, rng_state, restore_rng, atomic_checkpoint


def test_author_files_unchanged_and_import_adapter_is_minimal():
    originals = verify_originals()
    cls = load_author_model()
    module = sys.modules[cls.__module__]
    assert module.ADAPTER_AUDIT['source_sha256'] == originals['model.py']['sha256']
    assert module.ADAPTER_AUDIT['executed_sha256'] != originals['model.py']['sha256']


def test_author_objective_matches_three_view_weighted_loss_and_gradients():
    model = load_author_model()(8, 8, hidden_1024=16, out_1024=16, num_experts=18,
                                dropout=0, attn_drop=0, w12=.5, w13=.1, w23=.5)
    graph = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]))
    loss, result = model(graph, torch.randn(4, 8), torch.randn(4, 8), return_embeddings=True)
    torch.testing.assert_close(loss, .5 * result['loss12'] + .1 * result['loss13'] + .5 * result['loss23'])
    loss.backward()
    for prefix in ('gat_t5vit', 'gat_clip', 'fuse_gate', 'fuse_mlp.text', 'fuse_mlp.image', 'fuse_mlp.fusion'):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for name, parameter in model.named_parameters() if name.startswith(prefix))


def test_ppr_neighbors_match_closed_form_tiny_graph():
    graph = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=4)
    data = GraphData(graph, torch.empty(4, 0), None, {}, {'raw_sha256': {'test': 'closed_form_cycle_v1'}})
    config = {'alpha': .15, 'epsilon': 1e-9, 'top_k': 2, 'max_pushes': 10000,
              'seed_batch_size': 1, 'seed_limit': 4, 'direction': 'official_directed'}
    sampler = PPRSampler(data, config)
    nodes, residual = sampler.neighbors(0)
    # Exact PPR on directed 0->1->2->0 has p0 > p1 > p2; no path to 3.
    assert nodes.tolist() == [0, 1]
    assert residual < 1e-7
    singleton, mass = sampler.neighbors(3)
    assert singleton.tolist() == [3]
    assert mass < 1e-7


def test_checkpoint_restores_next_random_values_and_optimizer_update(tmp_path):
    seed_everything(13)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.3), torch.nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    x = torch.randn(6, 4)

    def update(current_model, current_optimizer):
        current_optimizer.zero_grad()
        loss = current_model(x).square().mean()
        loss.backward()
        current_optimizer.step()
        return loss.detach()

    update(model, optimizer)
    path = tmp_path / 'last.pt'
    atomic_checkpoint(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state()})
    expected = update(model, optimizer)
    expected_parameters = [value.detach().clone() for value in model.parameters()]
    reloaded = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.3), torch.nn.Linear(8, 1))
    opt = torch.optim.Adam(reloaded.parameters(), lr=.01)
    saved = torch.load(path, weights_only=False)
    reloaded.load_state_dict(saved['model'])
    opt.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    actual = update(reloaded, opt)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for left, right in zip(reloaded.parameters(), expected_parameters):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_official_nc_splits_are_disjoint():
    if not (ROOT / 'data/raw/mm_graph/ele-fashion/split.pt').exists():
        pytest.skip('Official dataset not downloaded')
    data = load_dataset('ele-fashion', 'text_first_assumed')
    assert data.features.shape == (97766, 1024)
    assert not data.audit['layout_verified']
    assert set(data.splits) == {'train', 'val', 'test'}


def test_pilot_run_artifacts_are_consistent():
    candidates = list((ROOT / 'runs').glob('*pilot_ele_fashion_cuda*'))
    if not candidates:
        pytest.skip('Pilot not yet executed')
    directory = sorted(candidates)[-1]
    status = json.loads((directory / 'status.json').read_text())
    if status['status'] != 'completed_pilot':
        pytest.skip('Latest pilot not complete')
    rows = [json.loads(line) for line in (directory / 'logs/metrics.jsonl').read_text().splitlines()]
    assert rows and all(np.isfinite(row['loss']) for row in rows)
    assert all(abs(row['loss'] - (.5 * row['loss_tv'] + .1 * row['loss_tf'] + .5 * row['loss_vf'])) < 1e-5 for row in rows)
    checkpoint = torch.load(directory / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
    assert checkpoint['step'] == rows[-1]['step'] == status['steps']
    assert checkpoint['cursor'] == rows[-1]['seed_cursor']
    assert list((directory / 'tensorboard').glob('events.*'))
    for name in ('training_loss', 'training_resources', 'embedding_pca', 'expert_usage'):
        assert (directory / 'figures' / (name + '.png')).stat().st_size > 1000


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for exact inference comparison')
def test_layerwise_full_neighbor_inference_matches_author_full_graph():
    from came_repro.evaluate import encode_layers
    seed_everything(4)
    model = load_author_model()(8, 8, hidden_1024=16, out_1024=16, num_experts=18, dropout=0, attn_drop=0).cuda().eval()
    graph = model._prep_graph(dgl.graph(([0, 1, 2, 0, 3], [1, 2, 0, 3, 1]), num_nodes=5))
    features = torch.randn(5, 8)
    with torch.no_grad():
        expected = model.gat_t5vit(graph.to('cuda'), features.cuda()).cpu()
    actual = encode_layers(model.gat_t5vit, graph, features, batch_size=2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_official_lp_graph_uses_training_edges_only():
    if not (ROOT / 'data/raw/mm_graph/sports-copurchase/lp-edge-split.pt').exists():
        pytest.skip('Official LP dataset not downloaded')
    data = load_dataset('sports-copurchase', 'text_first_assumed')
    source, target = data.graph.edges()
    torch.testing.assert_close(source, torch.as_tensor(data.splits['train']['source_node'], dtype=torch.long))
    torch.testing.assert_close(target, torch.as_tensor(data.splits['train']['target_node'], dtype=torch.long))
    assert data.audit['splits']['train'] == data.graph.num_edges()


def test_probe_selects_validation_best_and_reports_saved_test_predictions():
    directories = sorted((ROOT / 'runs').glob('*probe_ele_fashion*'))
    if not directories:
        pytest.skip('No completed probe')
    directory = directories[-1]
    summary = json.loads((directory / 'summary.json').read_text())
    rows = [json.loads(line) for line in (directory / 'logs/metrics.jsonl').read_text().splitlines()]
    best = max(rows, key=lambda row: row['val_accuracy'])
    assert summary['selected_epoch'] == best['step']
    assert summary['validation_accuracy'] == best['val_accuracy']
    predictions = torch.load(directory / 'test_predictions.pt', weights_only=True)
    raw = ROOT / 'data/raw/mm_graph/ele-fashion'
    split = torch.load(raw / 'split.pt', weights_only=True)
    labels = torch.tensor(torch.load(raw / 'labels-w-missing.pt', weights_only=True))
    assert predictions['node_ids'].tolist() == split['test_idx']
    torch.testing.assert_close(predictions['label'], labels[predictions['node_ids']])
    accuracy = (predictions['prediction'] == predictions['label']).double().mean().item()
    assert accuracy == summary['test_accuracy']
    assert summary['test_evaluations'] == 1 and not summary['paper_reproduction_complete']


def test_local_clipping_caps_applied_gradients_on_real_subgraphs():
    directories = sorted((ROOT / 'runs').glob('*local_ele_fashion_cuda*'))
    if not directories:
        pytest.skip('Local resource test not run')
    directory = directories[-1]
    config = json.loads((directory / 'config.json').read_text())
    if config['training'].get('max_grad_norm') is None:
        pytest.skip('No clipped local test')
    rows = [json.loads(line) for line in (directory / 'logs/metrics.jsonl').read_text().splitlines()]
    assert rows and rows[0]['grad_norm'] > 1e10 and rows[0]['zero_image_embeddings'] >= 1
    assert all(row['applied_grad_norm'] <= config['training']['max_grad_norm'] + 1e-5 for row in rows)
    assert all(np.isfinite(row['loss']) for row in rows)


def test_uncapped_large_gradient_control_fails_before_optimizer_update():
    directories = sorted((ROOT / 'runs').glob('*diagnostic_gradient_guard*'))
    if not directories:
        pytest.skip('Guard control not run')
    directory = directories[-1]
    status = json.loads((directory / 'status.json').read_text())
    assert status['status'] == 'failed' and status['steps'] == 0
    assert 'Large finite gradient' in status['error']
    assert not (directory / 'checkpoints/last.pt').exists()


def test_lp_audit_distinguishes_directed_disjointness_from_reverse_overlap():
    import importlib.util
    spec = importlib.util.spec_from_file_location('dataset_audit_test', ROOT / 'scripts/audit_datasets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    splits = {'train': {'source_node': [0, 1], 'target_node': [1, 2]},
              'valid': {'source_node': [1, 3], 'target_node': [0, 4]},
              'test': {'source_node': [2, 4], 'target_node': [1, 0]}}
    result = module.lp_overlaps(splits, 5)
    assert result['valid'] == {'ordered_train_overlap': 0, 'undirected_train_overlap': 1}
    assert result['test'] == {'ordered_train_overlap': 0, 'undirected_train_overlap': 1}
