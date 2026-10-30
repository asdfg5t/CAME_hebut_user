import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def test_validation_gate_ignores_test_and_paper_proximity():
    judge = module('run_priority_cloth_wn_validation').validation_passes
    assert judge({'val': .71, 'test': 0.}, .70, 'val')
    assert not judge({'val': .70, 'test': 1.}, .70, 'val')
    assert not judge({'val': .704, 'test': 1.}, .70, 'val')


def test_old_and_new_reference_cache_schemas_agree(tmp_path):
    resolve = module('priority_refinement_backend').reference_cache
    cache = str(tmp_path / 'cache')
    assert resolve({}, {'evaluation': {'embedding_cache': cache}}) == Path(cache)
    assert resolve({'embedding_cache': cache}, {}) == Path(cache)
    assert resolve({'embedding_cache': cache}, {'evaluation': {'embedding_cache': cache}}) == Path(cache)
    with pytest.raises(ValueError): resolve({'embedding_cache': cache}, {'evaluation': {'embedding_cache': str(tmp_path / 'other')}})
    with pytest.raises(ValueError): resolve({}, {})


def test_checkpoint_is_always_saved_on_pause_early_stop_or_final_epoch():
    due = module('priority_refinement_backend').checkpoint_due
    assert not due('wn', 1, 2000, 0, 100, False)
    assert due('wn', 25, 2000, 0, 100, False)
    assert due('cloth', 5, 80, 0, 20, False)
    assert due('wn', 3, 3, 0, 100, False)
    assert due('wn', 3, 2000, 100, 100, False)
    assert due('cloth', 2, 80, 0, 20, True)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_validation_refused(value):
    with pytest.raises(ValueError): module('run_priority_cloth_wn_validation').validation_passes({'val': value}, .7, 'val')


def test_partial_or_changed_resume_refused_before_loading_model(tmp_path):
    import torch
    backend = module('priority_refinement_backend')
    path = tmp_path / 'checkpoint.pt'
    saved = {'schema': 'came_priority_refinement_v1', 'task': 'wn', 'seed': 0,
             'config': {'identity': 1}, 'data_audit': {}, 'completed_epochs': 2, 'next_epoch': 3, 'history_rows': [{}]}
    torch.save(saved, path)
    with pytest.raises(ValueError): backend.restore_training(path, None, None, None, None, SimpleNamespace(config={'identity': 1}, data_audit={}), 'wn', 0)
    saved['history_rows'] = [{}, {}]; torch.save(saved, path)
    with pytest.raises(ValueError): backend.restore_training(path, None, None, None, None, SimpleNamespace(config={'identity': 2}, data_audit={}), 'wn', 0)


def test_lr_scheduler_resume_preserves_pending_plateau():
    import torch
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=1, threshold=0., min_lr=.00005)
    scheduler.step(1.); scheduler.step(1.)
    other_optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=.05)
    other_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(other_optimizer, factor=.1, patience=1, threshold=0., min_lr=.00005)
    other_optimizer.load_state_dict(optimizer.state_dict()); other_scheduler.load_state_dict(scheduler.state_dict())
    scheduler.step(1.); other_scheduler.step(1.)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(.005)
    assert other_optimizer.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
