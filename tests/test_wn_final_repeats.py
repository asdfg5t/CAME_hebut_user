import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_18_wn_final_repeats as wn


def profile():return json.loads((ROOT/'configs/wn_final_repeats_20261021.json').read_text())


@pytest.mark.parametrize('key,value',[('seeds',[0,1]),('nodes',1000),('width',512),('official_test_rows',100),('task','link_prediction'),('lr',.01),('max_new_seed0_tests',2),('metric_admission_gate',True)])
def test_wn_refuses_shortcuts_or_test_score_gating(key,value):
    p=profile();p[key]=value
    with pytest.raises(ValueError):wn.validate_profile(p)


def test_partial_summary_is_not_ten_seed_claim_and_keeps_lower_score(tmp_path,monkeypatch):
    monkeypatch.setattr(wn,'OUT',tmp_path)
    rows={'0':dict(seed=0,score_pct=78.,paper_pct=81.97,gap_pp=-3.97,official_test_queries=3134,run_directory='a'),
        '1':dict(seed=1,score_pct=77.,paper_pct=81.97,gap_pp=-4.97,official_test_queries=3134,run_directory='b')}
    wn.summarize(dict(state='running',results=rows),profile());value=json.loads((tmp_path/'results.json').read_text())
    assert value['statistics']['completed']==2 and value['statistics']['expected']==10
    assert value['statistics']['mean_pct']==77.5 and len(value['results'])==2
    assert value['statistics']['paper_reproduction_complete'] is False
    assert value['statistics']['metric_admission_gate'] is False
