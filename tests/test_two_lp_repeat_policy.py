"""Prevent delivering one training protocol while repeating a different protocol."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/run_verified_two_lp_repeats.py'
SPEC = importlib.util.spec_from_file_location('verified_two_lp_repeats', PATH)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def test_hard_delivery_cannot_silently_use_uniform_training():
    hard = {'negative_pool_size': 4, 'negative_policy': 'max_score_train_and_reverse_known_nonedge'}
    ENTRY.validate_training_policy('hard_four', hard)
    with pytest.raises(ValueError, match='differ'):
        ENTRY.validate_training_policy('uniform_one', hard)
    with pytest.raises(ValueError, match='differ'):
        ENTRY.validate_training_policy('hard_four', {})
    with pytest.raises(ValueError, match='filter'):
        ENTRY.validate_training_policy('hard_four', {**hard, 'negative_policy': 'exclude_validation_edges'})


def test_dispatch_retains_backend_protocol_and_exact_fixed_seeds():
    reference = Path('unchanged_validation_run')
    hard = ENTRY.child_arguments('hard_four', 'cloth-copurchase', 7, reference)
    uniform = ENTRY.child_arguments('uniform_one', 'cloth-copurchase', 7, reference)
    assert hard[0] == 'scripts/run_lp_hard_negative_candidate.py'
    assert '--train-one' not in hard and hard[hard.index('--seed') + 1] == '7'
    assert uniform[0] == 'scripts/run_locked_lp_repeats.py' and '--train-one' in uniform
    for seed in (0, -1, 10):
        with pytest.raises(ValueError, match='seed0'):
            ENTRY.child_arguments('hard_four', 'cloth-copurchase', seed, reference)
