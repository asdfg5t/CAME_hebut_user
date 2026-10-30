"""Mathematical objective and scientific selection invariants."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from came_repro.cloth_pairwise import pairwise_loss
from run_09_cloth_pairwise_validation import validate_plan, passes_gate, read, PROFILE


def test_pairwise_equation_gradient_and_common_shift():
    positive = torch.tensor([1., -2., 4.], dtype=torch.float64, requires_grad=True)
    negative = torch.tensor([0., 3., -1.], dtype=torch.float64, requires_grad=True)
    loss = pairwise_loss(positive, negative)
    torch.testing.assert_close(loss, -torch.nn.functional.logsigmoid(positive-negative).mean())
    torch.testing.assert_close(loss, pairwise_loss(positive+17, negative+17))
    loss.backward()
    expected = torch.sigmoid(negative-positive)/3
    torch.testing.assert_close(negative.grad, expected)
    torch.testing.assert_close(positive.grad, -expected)
    assert (negative.grad > 0).all() and (positive.grad < 0).all()


def test_improving_ranking_decreases_loss_and_extremes_finite():
    a = torch.tensor([-1000., 0., 1000.], requires_grad=True)
    b = -a.detach()
    old = pairwise_loss(a, b)
    assert pairwise_loss(a+1, b-1) < old
    old.backward()
    assert torch.isfinite(old) and torch.isfinite(a.grad).all()


def test_reject_unpaired_shapes():
    with pytest.raises(ValueError): pairwise_loss(torch.zeros(2), torch.zeros(2, 1))


def test_predeclared_plan_and_threshold():
    plan = read(PROFILE); validate_plan(plan)
    reference = plan['reference_validation_mrr']
    assert not passes_gate(plan, reference+.00499)
    assert passes_gate(plan, reference+.005)
    assert not passes_gate(plan, reference-.01)
    with pytest.raises(ValueError): passes_gate(plan, float('nan'))


@pytest.mark.parametrize('key,value', [('seed', 1), ('candidate_count', 2), ('formal_repeats', True),
    ('new_test_limit', 2), ('minimum_validation_gain_pp', .01), ('maximum_active_hours', 100), ('negative_pool_size', 8)])
def test_reject_undeclared_search_changes(key, value):
    plan = copy.deepcopy(read(PROFILE)); plan[key] = value
    with pytest.raises(ValueError): validate_plan(plan)


def test_reject_head_or_objective_replacement():
    for key, value in [('loss_reduction', 'bce'), ('max_epochs', 10000), ('lr', .05)]:
        plan = copy.deepcopy(read(PROFILE)); plan['recipe'][key] = value
        with pytest.raises(ValueError): validate_plan(plan)


def test_training_difference_is_explicit_and_original_intact():
    import hashlib
    original = ROOT/'scripts/run_lp_hard_negative_candidate.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == read(PROFILE)['derived_trainer_sha256']
    source = (ROOT/'src/came_repro/cloth_pairwise.py').read_text(encoding='utf-8')
    assert 'loss=pairwise_loss(positive,neg)' in source
    assert 'binary_cross_entropy_with_logits' not in source
    assert "sampler=TrainOnlyNegatives(np.concatenate((source,target)),np.concatenate((target,source)),len(gpu))" in source
