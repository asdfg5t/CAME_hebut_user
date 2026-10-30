import pytest
from came_repro.remaining_kg_validation import select_candidate
from came_repro.kg_selected_confirmation import validate_lock
from came_repro.author_loader import fingerprint


def test_selection_uses_validation_and_declared_gain_only():
    raw = {'summary':{'validation_accuracy':.8,'test_accuracy':.99}}
    small = {'summary':{'validation_accuracy':.804,'test_accuracy':1.}}
    better = {'summary':{'validation_accuracy':.81,'test_accuracy':.1}}
    assert select_candidate([small],raw) is raw
    assert select_candidate([small,better],raw) is better


def test_confirmation_binds_selected_head_and_held_out_test(tmp_path):
    best=tmp_path/'best.pt';best.write_bytes(b'fixed validation selected weights')
    summary={'dataset':'WN18RR','encoder_sha256':'encoder','graph_policy':'train_reverse',
             'feature_transform':'train_pair_affine','test_evaluations':0,'validation_accuracy':.82}
    lock={**summary,'classifier_sha256':fingerprint(best),'locked_before_test':True,'test_used_for_selection':False,
          'selected_validation_accuracy':.82}
    validate_lock(lock,summary,best)
    with pytest.raises(ValueError):validate_lock({**lock,'test_used_for_selection':True},summary,best)
    with pytest.raises(ValueError):validate_lock(lock,{**summary,'test_evaluations':1},best)
    best.write_bytes(b'changed weights')
    with pytest.raises(ValueError):validate_lock(lock,summary,best)
