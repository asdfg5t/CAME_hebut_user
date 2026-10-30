import copy
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_17_cloth_final_repeats as final


def profile():return json.loads((ROOT/'configs/cloth_final_repeats_20261019.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('key,value',[('seeds',[0,1]),('nodes',1000),('width',512),('official_test_queries',100),('fixed_negative_candidates',50),('maximum_new_seed0_tests',1),('metric_admission_gate',True)])
def test_final_protocol_rejects_seed_filtering_data_shortcuts_and_retests(key,value):
    p=profile();p[key]=value
    with pytest.raises(ValueError):final.validate_profile(p)


def test_final_recipe_cannot_change_after_known_seed0_test():
    p=profile();final.validate_profile(p);p['recipe']['lr']=.0001
    with pytest.raises(ValueError):final.validate_profile(p)


def test_reporting_keeps_low_seed_and_uses_all_seeds_for_statistics(tmp_path,monkeypatch):
    monkeypatch.setattr(final,'OUT',tmp_path)
    ledger=dict(state='running',results={'0':dict(seed=0,score_pct=26.82418986676,gap_pp=-2.05581013324,within_two_pp=False,
        official_test_queries=97456,reverse_nonoverlap_mrr_pct=26.47,reused=True,run_directory='saved'),
        '1':dict(seed=1,score_pct=20.,gap_pp=-8.88,within_two_pp=False,official_test_queries=97456,
        reverse_nonoverlap_mrr_pct=19.,reused=False,run_directory='new')})
    final.summarize(ledger,profile());saved=json.loads((tmp_path/'results.json').read_text())
    assert saved['statistics']['completed']==2 and saved['statistics']['expected']==10
    assert saved['statistics']['mean_pct']==pytest.approx((26.82418986676+20)/2)
    assert saved['statistics']['metric_admission_gate'] is False
    assert len(saved['results'])==2 and saved['results'][1]['score_pct']==20.
    assert saved['statistics']['paper_reproduction_complete'] is False
