import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('post_sports', Path(__file__).resolve().parents[1] / 'scripts/run_post_sports_main_repairs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def ledger(state='completed_delivered_two_lp_repeats', completed=10, complete=True):
    return {'state': state, 'statistics': {'sports-copurchase': {
        'completed': completed, 'complete_repeat_grid': complete}}}


def test_requires_all_formal_seeds_and_final_state():
    assert module.sports_ready(ledger())
    assert not module.sports_ready(ledger(completed=9))
    assert not module.sports_ready(ledger(state='running'))
    assert not module.sports_ready(ledger(state='failed'))
    assert not module.sports_ready(ledger(state='paused'))


def test_single_seed_or_missing_grid_cannot_release_diagnostics():
    assert not module.sports_ready(ledger(completed=1))
    assert not module.sports_ready(ledger(complete=False))
    assert not module.sports_ready(ledger(complete=1))
    assert not module.sports_ready({})
