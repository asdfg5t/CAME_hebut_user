"""Finite validation selection must never become a test-driven search."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from cloth_wn_autonomous import validate_profile,select_validation,require_all_validation_complete
from cloth_wn_autonomous_backend import candidate_by_id,configured_probe


@pytest.fixture
def plan():return json.loads((ROOT/'configs/cloth_wn_autonomous_20261017.json').read_text())


def test_fixed_finite_grid(plan):
    validate_profile(plan)
    for task in ('cloth','wn'):
        for candidate in plan[task+'_candidates']:
            kind,saved=candidate_by_id(candidate['id'])
            assert kind==task and saved==candidate


@pytest.mark.parametrize('key,value',[('seed',1),('formal_repeats',True),('offline',False),
    ('maximum_new_test_evaluations_per_task',2),('minimum_validation_gain_pp',.1),('maximum_active_hours',999)])
def test_reject_protocol_expansion(plan,key,value):
    plan[key]=value
    with pytest.raises(ValueError):validate_profile(plan)


def records(plan,task,values):
    key='validation_mrr' if task=='cloth' else 'validation_accuracy'
    return {c['id']:{'seed':0,'test_evaluations':0,key:v} for c,v in zip(plan[task+'_candidates'],values)}


def test_choose_validation_and_keep_fixed_tie_order(plan):
    values=records(plan,'cloth',[.315,.32,.32])
    selected=select_validation(plan,'cloth',values)
    assert selected['candidate_id']=='cloth_lr0005' and selected['passes_validation_gate']
    for item in values.values():item['paper_gap_or_test_hint']=999
    assert select_validation(plan,'cloth',values)==selected


def test_no_test_if_gain_below_threshold(plan):
    selected=select_validation(plan,'cloth',records(plan,'cloth',[.303,.304,.305]))
    assert not selected['passes_validation_gate']


def test_no_early_choice_with_missing_candidate(plan):
    values=records(plan,'wn',[.9,.9,.9]);values.pop('wn_lr0005')
    with pytest.raises(ValueError):select_validation(plan,'wn',values)


def test_never_use_a_tested_candidate(plan):
    values=records(plan,'wn',[.8,.81,.82]);values['wn_lr001']['test_evaluations']=1
    with pytest.raises(ValueError):select_validation(plan,'wn',values)


def test_both_tasks_finish_all_validation_before_any_test(plan):
    jobs={'validate_'+c['id']:{'state':'completed'} for task in ('cloth','wn') for c in plan[task+'_candidates']}
    require_all_validation_complete(plan,jobs)
    jobs['validate_wn_lr0005']['state']='running'
    with pytest.raises(ValueError):require_all_validation_complete(plan,jobs)


def test_learning_rate_reaches_real_trainer(monkeypatch):
    import came_repro.downstream as downstream
    calls=[]
    def stub(*args,**kwargs):calls.append(kwargs['lr']);return 'ok'
    monkeypatch.setattr(downstream,'train_linear_probe',stub)
    assert configured_probe({'lr':.01},object())=='ok'
    assert calls==[.01]
