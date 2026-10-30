"""Budget-only handoff must not silently resume a different protocol."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from came_repro.goodreads_official import resume_configuration_matches
from run_21_goodreads_official import config_for as original_config
from run_23_goodreads_four_rounds import config_for


def inputs():
    return [json.loads((ROOT / 'configs' / name).read_text()) for name in
        ['goodreads_official_rebuild_20261021.json', 'goodreads_four_rounds_20261021.json']]


def test_only_optimization_round_budget_changes_and_no_input_mutation(monkeypatch):
    from came_repro import author_loader
    monkeypatch.setattr(author_loader, 'fingerprint', lambda path: 'unit-test-release')
    p, short = inputs()
    snapshot = copy.deepcopy(p)
    a, b = original_config(p, 3), config_for(p, short, 3)
    assert p == snapshot
    assert a['seed'] == b['seed'] == 3
    old, new = a['evaluation']['recipe'], b['evaluation']['recipe']
    assert old['max_rounds'] == 30 and new['max_rounds'] == 4
    assert {k:v for k,v in old.items() if k!='max_rounds'} == {k:v for k,v in new.items() if k!='max_rounds'}
    assert a['evaluation']['embedding_cache'] == b['evaluation']['embedding_cache']
    assert not b['evaluation']['test_used_for_selection']


def test_four_round_resume_accepts_identical_config_but_rejects_thirty_rounds(monkeypatch):
    from came_repro import author_loader
    monkeypatch.setattr(author_loader, 'fingerprint', lambda path: 'unit-test-release')
    p, short = inputs()
    old, new = original_config(p, 0), config_for(p, short, 0)
    assert resume_configuration_matches(new, copy.deepcopy(new))
    assert not resume_configuration_matches(old, new)
    changed = copy.deepcopy(new)
    changed['evaluation']['recipe']['max_rounds'] = 5
    assert not resume_configuration_matches(new, changed)


def test_short_budget_keeps_all_seeds_and_official_scoring():
    _, short = inputs()
    assert short['seeds'] == list(range(10))
    assert short['official_validation_queries'] == 570464
    assert short['official_test_queries'] == 1710677
    assert short['fixed_negatives'] == 150 and short['cache_width'] == 1024
    assert not short['test_score_admission_gate']
    assert not short['formal_paper_protocol_equivalence_claimed']
