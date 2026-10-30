"""Academic invariants for the reconstructed multi-graph protocol and exact inference."""
from types import SimpleNamespace

import dgl
import numpy as np
import pytest
import torch

from came_repro.author_loader import load_author_model
from came_repro.data import GraphData, PPRSampler
from came_repro.joint_sampling import JointSchedule, PackedPPRSampler
from came_repro.stream_inference import batch_full_neighbor_embeddings


def test_wiki_selection_uses_stopping_mask_and_preserves_validation():
    from came_repro.downstream import node_inputs
    data=SimpleNamespace(audit={'dataset':'wiki-cs'},labels=np.array([0,1,0,1]),
                         splits={'train_masks':np.array([[1,0,0,0]]),'val_masks':np.array([[0,1,0,0]]),
                                 'stopping_masks':np.array([[0,0,1,0]]),'test_mask':np.array([0,0,0,1])})
    train,stop,test=node_inputs(data,np.zeros((4,8)),0)
    assert train.indices.tolist()==[0]
    assert stop.indices.tolist()==[2]
    assert test.indices.tolist()==[3]
    assert data.splits['val_masks'][0].tolist()==[0,1,0,0]


def test_joint_epoch_covers_every_node_once_in_each_graph():
    counts = {'products': 17, 'books': 5, 'cloth': 8}
    schedule = JointSchedule(counts, 4, torch.Generator().manual_seed(42))
    visited = {name: [] for name in counts}
    graph_order = []
    while True:
        selected = schedule.next_batch()
        if selected is None:
            break
        name, seeds = selected
        graph_order.append(name)
        visited[name].extend(seeds.tolist())
    assert graph_order[:3] == list(counts)
    assert all(sorted(visited[name]) == list(range(count)) for name, count in counts.items())
    assert schedule.cursors == counts


def test_joint_checkpoint_preserves_next_batch_and_next_epoch():
    counts = {'products': 19, 'books': 7, 'cloth': 11}
    schedule = JointSchedule(counts, 3, torch.Generator().manual_seed(9))
    for _ in range(5):
        schedule.next_batch()
    state = schedule.state_dict()
    clone = JointSchedule(counts, 3, torch.Generator().manual_seed(123))
    clone.load_state_dict(state)
    while True:
        left, right = schedule.next_batch(), clone.next_batch()
        assert (left is None) == (right is None)
        if left is None:
            break
        assert left[0] == right[0]
        torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)
    schedule.new_epoch()
    clone.new_epoch()
    for name in counts:
        torch.testing.assert_close(schedule.orders[name], clone.orders[name], rtol=0, atol=0)


def test_packed_ppr_preserves_original_transition_rule(tmp_path, monkeypatch):
    import came_repro.joint_sampling as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    graph = dgl.graph(([0, 0, 1, 2, 2], [1, 1, 2, 0, 3]), num_nodes=5)
    data = GraphData(graph, torch.empty(5, 0), None, {}, {'raw_sha256': {'fixture': 'weighted_cycle_ppr_v1'}})
    config = {'alpha': .15, 'epsilon': 1e-8, 'top_k': 4, 'max_pushes': 100000, 'seed_batch_size': 1, 'seed_limit': 5, 'direction': 'official_directed'}
    original = PPRSampler(data, config)
    packed = PackedPPRSampler(data, config)
    try:
        for seed in (0, 1, 4):
            expected, mass = original.neighbors(seed)
            actual, actual_mass = packed.neighbors(seed)
            np.testing.assert_array_equal(actual, expected)
            assert actual_mass == pytest.approx(mass, abs=1e-12)
        packed.flush()
        nodes, _ = packed.neighbors(0)
        np.testing.assert_array_equal(nodes, original.neighbors(0)[0])
    finally:
        packed.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_exact_two_hop_inference_matches_preserved_author_full_graph():
    torch.manual_seed(21)
    model = load_author_model()(8, 8, hidden_1024=16, out_1024=16, num_experts=18, dropout=0, attn_drop=0).cuda().eval()
    graph = dgl.graph(([0, 1, 2, 0, 3], [1, 2, 0, 3, 1]), num_nodes=5)
    text, vision = torch.randn(5, 8), torch.randn(5, 8)
    data = SimpleNamespace(graph=graph, modality_batch=lambda index: (text[index], vision[index]))
    with torch.no_grad():
        _, expected = model(graph.to('cuda'), text.cuda(), vision.cuda(), return_embeddings=True, compute_loss=False)
    graph = model._prep_graph(graph)
    loader = dgl.dataloading.DataLoader(graph, torch.arange(5), dgl.dataloading.MultiLayerFullNeighborSampler(2), batch_size=2, shuffle=False, num_workers=0)
    actual = torch.empty(5, 16)
    for inputs, targets, blocks in loader:
        actual[targets] = batch_full_neighbor_embeddings(model, data, blocks, inputs).cpu()
    torch.testing.assert_close(actual, expected['z3'].cpu(), rtol=1e-5, atol=1e-6)
