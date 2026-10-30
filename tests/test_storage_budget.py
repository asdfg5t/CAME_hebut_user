import json
import pytest
import torch

from came_repro import storage_budget


def test_cache_allocation_refuses_high_water_even_with_disk_free(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    (root / 'configs').mkdir(parents=True)
    cache = tmp_path / 'cache'
    cache.mkdir()
    config = {'large_checkpoint_root': str(cache / 'checkpoints'), 'embedding_cache_root': str(cache / 'embeddings'),
              'project_high_water_gib': 1, 'external_cache_high_water_gib': 1, 'reserve_free_gib': 0}
    (root / 'configs/storage.json').write_text(json.dumps(config))
    monkeypatch.setattr(storage_budget, 'ROOT', root)
    assert storage_budget.check_budget()['within_budget']
    with pytest.raises(RuntimeError, match='Storage budget exceeded'):
        storage_budget.guard_allocation(cache / 'embeddings/large.npy', 2 * 2**30)


def test_checkpoint_budget_counts_shared_storage_once():
    tensor = torch.zeros(1000)
    first = storage_budget.tensor_storage_bytes({'model': tensor})
    shared = storage_budget.tensor_storage_bytes({'model': tensor, 'view': tensor[:10]})
    independent = storage_budget.tensor_storage_bytes({'model': tensor, 'copy': tensor.clone()})
    assert first == shared
    assert independent > shared
