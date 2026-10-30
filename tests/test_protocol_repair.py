import copy
import numpy as np
import pytest
from came_repro.protocol_repair import bidirected_train_keys
from came_repro.all_unmet_repair_pipeline import validation_improves


def test_train_derivation_keeps_ids_and_removes_parallel_and_self_edges():
    source=np.array([0,0,1,2]);target=np.array([1,1,0,2])
    keys=bidirected_train_keys(source,target,4)
    np.testing.assert_array_equal(keys,np.array([1,4]))
    assert 0*4+3 not in keys and 3*4+0 not in keys
    with pytest.raises(ValueError):bidirected_train_keys(np.array([4]),np.array([0]),4)


def test_validation_gate_does_not_read_or_select_test_scores():
    candidate={'summary':{'validation_mrr':.326,'test':{'official':{'mrr':.01}}}}
    reference={'summary':{'validation_mrr':.32,'test':{'official':{'mrr':.99}}}}
    assert validation_improves(candidate,reference,'validation_mrr',.5)
    changed=copy.deepcopy(candidate);changed['summary']['test']['official']['mrr']=1.
    assert validation_improves(changed,reference,'validation_mrr',.5)
    candidate['summary']['validation_mrr']=.324
    assert not validation_improves(candidate,reference,'validation_mrr',.5)
    candidate['summary']['validation_mrr']=float('nan')
    with pytest.raises(ValueError):validation_improves(candidate,reference,'validation_mrr',.5)
