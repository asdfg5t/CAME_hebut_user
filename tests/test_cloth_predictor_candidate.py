"""Prevent a predictor-structure experiment from silently changing its control recipe."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('cloth_predictor_candidate',
    Path(__file__).resolve().parents[1] / 'scripts/run_cloth_predictor_candidate.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def control():
    return dict(head_variant='existing_two_layer_head_upstream_schedule', lr=.001,
                batch_size=4096, max_epochs=300, patience=10, step_lr_size=None,
                step_lr_gamma=None, loss_reduction='separate_mean_sum', seed=0, negative_pool_size=4)


def test_only_predictor_structure_changes_all_other_controls_are_retained():
    original = control()
    recipe = MODULE.structure_recipe(original)
    assert recipe['head_variant'] == 'two_hidden_layernorm_head_upstream_schedule'
    assert {k: recipe[k] for k in MODULE.RECIPE_KEYS if k != 'head_variant'} == {
        k: original[k] for k in MODULE.RECIPE_KEYS if k != 'head_variant'}
    assert original == control()


@pytest.mark.parametrize('key,value', [('seed', 3), ('negative_pool_size', 1),
    ('head_variant', 'two_hidden_layernorm_head_upstream_schedule')])
def test_refuse_incompatible_reference(key, value):
    changed = control()
    changed[key] = value
    with pytest.raises(ValueError):
        MODULE.structure_recipe(changed)
