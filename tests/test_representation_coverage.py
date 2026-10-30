"""Protect the one-factor protocol, historical PPR caches and exact pool prefix."""
import copy
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from representation_coverage_backend import validate_single_change, clone_readonly_cache


def candidate():
    old = json.loads((ROOT / 'configs/subset_offline_24h.json').read_text())
    new = copy.deepcopy(old)
    new['name'] = 'coverage_candidate'
    new['sampling']['seed_limits'] = {k: 2*v for k, v in old['sampling']['seed_limits'].items()}
    return old, new


def test_only_uniform_pool_size_changes():
    old, new = candidate()
    validate_single_change(old, new)


@pytest.mark.parametrize('section,key,value', [('model', 'hidden_dim', 512),
    ('sampling', 'epsilon', .00001), ('training', 'epochs', 16),
    ('training', 'max_steps', 100), ('training', 'lr', .0005)])
def test_rejects_confounding_changes(section, key, value):
    old, new = candidate(); new[section][key] = value
    with pytest.raises(ValueError): validate_single_change(old, new)


def test_clone_preserves_every_historical_row_and_rejects_changed_copy(tmp_path):
    source = tmp_path / 'original/old.sqlite3'; source.parent.mkdir()
    with sqlite3.connect(str(source)) as db:
        db.execute('CREATE TABLE ppr(seed INTEGER PRIMARY KEY,nodes BLOB NOT NULL,residual REAL NOT NULL)')
        db.executemany('INSERT INTO ppr VALUES(?,?,?)', [(9, b'aa', .125), (1, b'bb', .25)])
    before = source.read_bytes(); target = tmp_path / 'new/private.sqlite3'
    report = clone_readonly_cache(source, target)
    assert report['logical_rows'] == 2 and source.read_bytes() == before
    clone_readonly_cache(source, target)
    with sqlite3.connect(str(target)) as db: db.execute('UPDATE ppr SET nodes=? WHERE seed=1', (b'changed',))
    with pytest.raises(ValueError): clone_readonly_cache(source, target)
    assert source.read_bytes() == before


def test_clone_refuses_writing_inside_original_directory(tmp_path):
    source = tmp_path / 'historical.sqlite3'; source.touch()
    with pytest.raises(ValueError): clone_readonly_cache(source, tmp_path / 'unsafe/new.sqlite3')


def test_expanded_uniform_pool_has_original_prefix_without_labels(monkeypatch):
    import torch
    import came_repro.subset_offline_train as subset
    monkeypatch.setattr(subset, 'CURRENT_EXPERIMENT', None)
    counts = {'ogbn-products': 57, 'books-lp': 53, 'cloth-copurchase': 31}
    old = {'ogbn-products': 8, 'books-lp': 8, 'cloth-copurchase': 4}
    a = subset.FixedSubsetSchedule(counts, 2, torch.Generator().manual_seed(42), old)
    b = subset.FixedSubsetSchedule(counts, 2, torch.Generator().manual_seed(42), {k:2*v for k,v in old.items()})
    for k, n in old.items():
        assert torch.equal(a.pools[k], b.pools[k][:n]) and len(b.pools[k].unique()) == 2*n
