from pathlib import Path
import sys
import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from came_repro.compact_csc import build_index, load_graph
from came_repro.regularized_link_head import RegularizedLinkHead
from came_repro.lp_validation_repair import LayerNormLinkHead


def test_parallel_direction_loops_isolated_and_reuse(tmp_path):
    edges = np.array([[2, 1], [0, 1], [2, 1], [1, 1], [1, 1], [3, 0]])
    a = build_index(edges, 5, tmp_path, 'fixture', chunk_size=2)
    ptr = np.load(tmp_path/'indptr.npy'); values = np.load(tmp_path/'indices.npy')
    assert values.dtype == ptr.dtype == np.int32
    assert a['removed_selfloops'] == 2 and a['edges'] == 9
    assert values[ptr[1]:ptr[2]].tolist() == [2, 0, 2, 1]
    assert values[ptr[4]:ptr[5]].tolist() == [4]
    assert build_index(edges, 5, tmp_path, 'fixture') == a
    g = load_graph(tmp_path)
    assert g.idtype == torch.int32 and g.num_edges() == 9
    assert g.formats()['created'] == ['csc']
    with pytest.raises(ValueError): build_index(edges, 5, tmp_path, 'wrong_source')


@pytest.mark.parametrize('edges,n', [(np.array([[-1, 0]]), 2), (np.array([[0, 2]]), 2),
    (np.array([[0., 1.]]), 2), (np.array([[0, 1]]), 2**31)])
def test_invalid_ids_or_width_rejected(tmp_path, edges, n):
    with pytest.raises(ValueError): build_index(edges, n, tmp_path, 'fixture')


def test_empty_graph_gets_one_loop_per_node(tmp_path):
    build_index(np.empty((0, 2), dtype=np.int64), 4, tmp_path, 'empty')
    assert np.load(tmp_path/'indices.npy').tolist() == [0, 1, 2, 3]


def test_dropout_eval_identical_and_training_randomness():
    torch.manual_seed(4)
    original = LayerNormLinkHead(16); candidate = RegularizedLinkHead(16)
    candidate.load_state_dict(original.state_dict())
    a, b = torch.randn(10, 16), torch.randn(10, 16)
    original.eval(); candidate.eval()
    assert torch.equal(original(a, b), candidate(a, b))
    candidate.train(); assert not torch.equal(candidate(a, b), candidate(a, b))
    assert candidate.state_dict().keys() == original.state_dict().keys()


@pytest.mark.parametrize('field,value', [('seed', 1), ('formal_repeats', True),
    ('maximum_new_tests_per_task', 2), ('validation_gain_threshold_pp', .1), ('maximum_active_hours', 999)])
def test_fixed_all_task_plan_rejects_search_expansion(field, value):
    import copy
    from run_10_unresolved_main import validate_plan, read, PROFILE
    plan = copy.deepcopy(read(PROFILE)); plan[field] = value
    with pytest.raises(ValueError): validate_plan(plan)


def test_complete_official_data_and_validation_gate():
    import copy
    from run_10_unresolved_main import validate_plan, passes_gate, read, PROFILE
    plan = read(PROFILE); validate_plan(plan)
    summary = dict(seed=0, test_evaluations=0, validation_mrr=plan['cloth']['reference_validation']+.005)
    assert passes_gate(plan, 'cloth', summary)
    summary['validation_mrr'] -= .0001; assert not passes_gate(plan, 'cloth', summary)
    summary['test_evaluations'] = 1
    with pytest.raises(ValueError): passes_gate(plan, 'cloth', summary)
    changed = copy.deepcopy(plan); changed['products']['official_test_nodes'] = 1000
    with pytest.raises(ValueError): validate_plan(changed)


def test_products_writer_has_only_declared_changes():
    import ast
    original = (ROOT/'scripts/run_products_single_validation.py').read_text(encoding='utf-8')
    candidate = (ROOT/'scripts/unresolved_products_full_20261018.py').read_text(encoding='utf-8')
    def function(source):
        tree = ast.parse(source); node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'exact_inference')
        return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
    expected = function(original).replace(
        "    graph=model._prep_graph(data.graph);data.graph=None;gc.collect();graph=graph.formats('csc');data.graph=graph",
        '    graph=data.graph  # CSC certificate already implements exact author loop semantics.')
    expected = expected.replace('dtype=torch.long)', 'dtype=graph.idtype)').replace(
        '            pause_if_requested();cursor=', '            resource_guard();pause_if_requested();cursor=')
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(function(candidate)))
    from unresolved_products_full_20261018 import next_cursor
    assert next_cursor(np.arange(10, 14), 10, 20) == 14


def test_compatibility_adapter_casts_only_structural_ids():
    import dgl
    from run_11_products_block64_repair import block64_forward
    graph = dgl.graph((torch.tensor([0, 1], dtype=torch.int32), torch.tensor([1, 2], dtype=torch.int32)), num_nodes=3)
    inputs, targets, blocks = dgl.dataloading.MultiLayerFullNeighborSampler(2).sample_blocks(graph, torch.tensor([2], dtype=torch.int32))
    observed = []
    def original(model, data, converted, ids):
        assert all(block.idtype == torch.int64 for block in converted)
        assert ids.dtype == torch.int64
        for before, after in zip(blocks, converted):
            assert before.num_edges() == after.num_edges() and before.num_src_nodes() == after.num_src_nodes()
            for a, b in zip(before.edges(), after.edges()): assert torch.equal(a.long(), b)
        observed.append(True); return 'result'
    assert block64_forward(original)(None, None, blocks, inputs) == 'result'
    assert observed and all(block.idtype == torch.int32 for block in blocks)


def test_native_csc64_sampling_preserves_parallel_edges_and_node_order():
    import dgl
    from run_12_products_csc64_repair import native_csc64, validate_sampled_blocks
    graph = dgl.graph((torch.tensor([0,0,1,2,3], dtype=torch.int32),
                       torch.tensor([1,1,2,0,3], dtype=torch.int32)), num_nodes=5).formats('csc')
    candidate = native_csc64(graph)
    assert candidate.idtype == torch.int64 and graph.idtype == torch.int32
    assert candidate.num_nodes() == 5 and candidate.num_edges() == 5
    for old, new in zip(graph.adj_tensors('csc'), candidate.adj_tensors('csc')):
        assert torch.equal(old.long(), new)
    inputs, _, blocks = dgl.dataloading.MultiLayerFullNeighborSampler(2).sample_blocks(candidate, torch.tensor([2]))
    validate_sampled_blocks(blocks, inputs)
    with pytest.raises(ValueError, match='ordering'):
        validate_sampled_blocks(blocks, inputs.flip(0))
    with pytest.raises(ValueError, match='int64'):
        validate_sampled_blocks(blocks, inputs.int())
    blocks[0].dstdata[dgl.NID] = blocks[0].dstdata[dgl.NID].flip(0)
    with pytest.raises(ValueError, match='prefix'):
        validate_sampled_blocks(blocks, inputs)


def test_listwise_ranking_loss_is_shift_invariant_and_has_correct_gradients():
    from came_repro.listwise_link_train import listwise_loss
    scores = torch.tensor([[1.,0.,-1.,2.,.5], [.3,1.,2.,3.,4.]], requires_grad=True)
    loss = listwise_loss(scores)
    expected = (torch.logsumexp(scores,dim=1)-scores[:,0]).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(listwise_loss(scores+200), loss, rtol=1e-5, atol=1e-5)
    loss.backward()
    assert (scores.grad[:,0] < 0).all() and (scores.grad[:,1:] > 0).all()
    improved = scores.detach().clone(); improved[:,0] += 1
    assert listwise_loss(improved) < loss
    with pytest.raises(ValueError): listwise_loss(scores[:,:4])
