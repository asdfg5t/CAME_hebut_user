import copy
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_16_cloth_listwise_completion import validate_admission
from remaining_objective_backend_20261018 import verify_plan


def prior():
    p=verify_plan()
    return dict(records=dict(validate_cloth=dict(state='completed',summary=dict(seed=0,test_evaluations=0,
        validation_mrr=p['reference_validation']['cloth']+.005)),confirm_cloth=dict(state='completed')))


def test_only_predeclared_validation_gain_allows_confirmation_admission():
    value=prior();assert validate_admission(value)[0]
    value['records']['validate_cloth']['summary']['validation_mrr']-=.000001
    assert validate_admission(value)==(False,'validation_gain_not_met')


def test_partial_validation_or_partial_confirmation_cannot_start_repeats():
    value=prior();value['records']['validate_cloth']['state']='running'
    assert validate_admission(value)==(False,'validation_incomplete')
    value=prior();value['records']['confirm_cloth']['state']='running'
    assert validate_admission(value)==(False,'confirmation_incomplete')


@pytest.mark.parametrize('field,value',[('seed',1),('test_evaluations',1)])
def test_test_score_cannot_reclassify_candidate_validation(field,value):
    item=copy.deepcopy(prior());item['records']['validate_cloth']['summary'][field]=value
    with pytest.raises(ValueError):validate_admission(item)
