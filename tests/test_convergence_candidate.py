"""Meaningful budget-migration protections; no benchmark or CUDA training here."""
import copy
import importlib.util
from pathlib import Path
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'scripts/run_convergence_candidate.py'
spec=importlib.util.spec_from_file_location('convergence_candidate',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def states():
    old={'schema':'came_lp_protocol_v1','completed_epochs':100,'next_epoch':101,'stale':0,'config':{'name':'parent','evaluation':{'max_epochs':100,'patience':10,'lr':.001,'batch_size':4096}},'head':{'weight':torch.arange(4)},'optimizer':{'state':{0:{'exp_avg':torch.ones(4)}}},'rng':torch.get_rng_state(),'history_rows':[{'step':100,'validation_mrr':.2742}]}
    config=copy.deepcopy(old['config']);config['name']='new';config['evaluation'].update(max_epochs=300,convergence_profile_sha256='declared')
    return old,config


def test_budget_only_migration_preserves_trained_state_and_parent():
    old,config=states();new=module.migrated_lp_state(old,config,300)
    assert torch.equal(new['head']['weight'],old['head']['weight'])
    assert torch.equal(new['optimizer']['state'][0]['exp_avg'],old['optimizer']['state'][0]['exp_avg'])
    assert torch.equal(new['rng'],old['rng']) and new['history_rows']==old['history_rows']
    assert old['config']['evaluation']['max_epochs']==100 and new['config']['evaluation']['max_epochs']==300


@pytest.mark.parametrize('change',['lr','early_stopped','cursor','budget'])
def test_reject_silent_protocol_change_or_invalid_boundary(change):
    old,config=states()
    if change=='lr':config['evaluation']['lr']=.01
    elif change=='early_stopped':old['stale']=10
    elif change=='cursor':old['next_epoch']=100
    else:old['completed_epochs']=99;old['next_epoch']=100
    with pytest.raises(ValueError):module.migrated_lp_state(old,config,300)
