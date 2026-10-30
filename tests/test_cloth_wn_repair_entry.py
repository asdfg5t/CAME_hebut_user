"""Recovery must preserve committed optimizer/sampling identity and user pauses."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import cloth_wn_repair_entry as entry


@pytest.fixture
def checkpoint():
    config = json.loads((ROOT/'configs/representation_coverage_20261016.json').read_text())
    names = config['datasets']; limits = config['sampling']['seed_limits']
    pools = {n: torch.arange(limits[n]) for n in names}
    state = {'schema':'came_joint_v1', 'config':copy.deepcopy(config),
        'step':13500, 'completed_epochs':0, 'completed_subset_rounds':5,
        'model':{'weight':torch.tensor([.5])},
        'optimizer':{'state':{0:{'exp_avg':torch.tensor([.1]),'exp_avg_sq':torch.tensor([.01])}},'param_groups':[{'lr':.0001}]},
        'rng':{'python':(), 'numpy':(), 'torch':torch.tensor([1],dtype=torch.uint8),
               'cuda':[torch.tensor([1],dtype=torch.uint8)]},
        'schedule':{'epoch':5, 'seed_limits':limits, 'batch_size':32,
                    'orders':{n:p.clone() for n,p in pools.items()},
                    'fixed_seed_pools':pools,
                    'cursors':dict(zip(names,[7488,7456,7456]))},
        'counts':{n:{'steps':s,'skipped':0} for n,s in zip(names,[5354,5353,2793])},
        'data_audit':{}}
    return state,config,pools


def test_committed_step_not_latest_printed_log(checkpoint):
    result=entry.validate_checkpoint(*checkpoint)
    assert result['next_optimizer_step']==13501
    assert result['remaining_optimizer_updates']==6980


@pytest.mark.parametrize('change', ['step', 'pool', 'order', 'cursor', 'adam', 'cuda_rng', 'nonfinite', 'full_epoch'])
def test_corrupt_or_nonmatching_state_is_rejected(checkpoint,change):
    state,config,pools=checkpoint
    if change=='step':state['step']=13672
    elif change=='pool':state['schedule']['fixed_seed_pools']=copy.deepcopy(pools);state['schedule']['fixed_seed_pools']['books-lp'][0]=-1
    elif change=='order':state['schedule']['orders']['books-lp'][0]=-1
    elif change=='cursor':state['schedule']['cursors']['books-lp']=7457
    elif change=='adam':state['optimizer']['state']={}
    elif change=='cuda_rng':state['rng']['cuda']=[]
    elif change=='nonfinite':state['model']['weight'][0]=float('nan')
    elif change=='full_epoch':state['completed_epochs']=5
    with pytest.raises(ValueError):entry.validate_checkpoint(state,config,pools)


def test_changed_scientific_protocol_is_rejected(checkpoint):
    state,config,pools=checkpoint
    config['training']['lr']=.001
    with pytest.raises(ValueError,match='configuration'):
        entry.validate_checkpoint(state,config,pools)


def test_pause_is_not_removed_by_check_only(tmp_path,monkeypatch,capsys):
    import came_repro.ide_workflow as workflow
    class Guard:
        def close(self):pass
    monkeypatch.setattr(workflow,'acquire_gpu_lock',lambda *a:Guard())
    marker=tmp_path/'pause.json';marker.write_text('{"reason":"user"}')
    monkeypatch.setattr(entry,'OUT',tmp_path/'out');monkeypatch.setattr(entry,'PAUSE',marker)
    monkeypatch.setattr(entry,'audit_preflight',lambda *a:pytest.fail('Paused training must not be audited or started'))
    assert entry.main(['--check-only'])==0
    assert marker.read_text()=='{"reason":"user"}'
    assert '暂停' in capsys.readouterr().out


def test_busy_entry_starts_no_second_job(tmp_path,monkeypatch,capsys):
    import came_repro.ide_workflow as workflow
    def busy(*args):raise RuntimeError('busy')
    monkeypatch.setattr(workflow,'acquire_gpu_lock',busy)
    monkeypatch.setattr(entry,'OUT',tmp_path/'out')
    monkeypatch.setattr(entry,'audit_preflight',lambda *a:pytest.fail('Do not start while busy'))
    assert entry.main([])==0
    assert '重复' in capsys.readouterr().out


def test_resume_archives_only_this_task_pause(tmp_path,monkeypatch):
    marker=tmp_path/'this_pause.json';marker.write_bytes(b'{"reason":"user"}')
    unrelated=tmp_path/'products_pause.json';unrelated.write_bytes(b'protected')
    monkeypatch.setattr(entry,'OUT',tmp_path/'out');monkeypatch.setattr(entry,'PAUSE',marker)
    entry.clear_explicit_pause()
    archived=list((tmp_path/'out/pause_history').glob('*.json'))
    assert len(archived)==1 and archived[0].read_bytes()==b'{"reason":"user"}'
    assert not marker.exists() and unrelated.read_bytes()==b'protected'


def test_resume_does_not_clear_a_newer_pause(tmp_path,monkeypatch):
    marker=tmp_path/'pause.json';marker.write_bytes(b'old request')
    expected=entry.digest(marker)
    marker.write_bytes(b'newer user request')
    monkeypatch.setattr(entry,'PAUSE',marker);monkeypatch.setattr(entry,'OUT',tmp_path/'out')
    with pytest.raises(RuntimeError,match='newer'):
        entry.clear_explicit_pause(expected)
    assert marker.read_bytes()==b'newer user request'
