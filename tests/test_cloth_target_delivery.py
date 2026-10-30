import copy
import json
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import cloth_target_delivery as delivery


def profile():
    return json.loads((ROOT/'configs/cloth_target_delivery_20261018.json').read_text(encoding='utf-8'))


def test_gap_gate_uses_absolute_percentage_points_and_never_rounds():
    p=profile()
    assert delivery.within_target(26.88,p)
    assert not delivery.within_target(26.879999,p)
    assert not delivery.within_target(26.810958023732056,p)
    assert delivery.within_target(28.88,p)
    assert not delivery.within_target(30.880001,p)
    with pytest.raises(ValueError): delivery.within_target(float('nan'),p)


@pytest.mark.parametrize('field,value',[
    ('seeds',[0,1]),('nodes',1000),('official_test_queries',100),
    ('fixed_negative_candidates',50),('maximum_new_seed0_confirmations',2),
    ('no_automatic_strategy_search_after_test',False),('prior_diagnostic_half_pp_gate_not_reclassified',False)])
def test_delivery_refuses_shortcuts_and_test_driven_strategy_search(field,value):
    p=copy.deepcopy(profile());p[field]=value
    with pytest.raises(ValueError): delivery.validate_profile(p)


def test_new_lock_cannot_replace_already_selected_head(tmp_path,monkeypatch):
    from came_repro.author_loader import fingerprint
    run=tmp_path/'run';cache=tmp_path/'cache';cp=tmp_path/'cp';out=tmp_path/'out'
    for directory in [run/'checkpoints',cache,cp,out]: directory.mkdir(parents=True)
    def write(path,value): path.write_text(json.dumps(value),encoding='utf-8')
    write(cp/'link_head_best.pt',{'head':'fixed-before-test'})
    write(cache/'inference_state.json',{'complete':True})
    write(run/'checkpoints/location.json',{'checkpoint_directory':str(cp)})
    write(run/'summary.json',{'seed':0,'head_variant':'fixed','embedding_cache':str(cache),'validation_mrr':.3})
    monkeypatch.setattr(delivery,'OUT',out)
    lock=delivery.test_lock(run,profile())
    before=fingerprint(lock)
    write(cp/'link_head_best.pt',{'head':'changed-after-test'})
    with pytest.raises(ValueError,match='不得替换'): delivery.test_lock(run,profile())
    assert fingerprint(lock)==before
