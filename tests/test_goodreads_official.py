"""Scientific boundaries: official ties/candidates, budget and withdrawn simulation."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from came_repro.goodreads_official import official_ranks,metrics,make_head,pair_logits


def test_official_tie_ranks_and_full_150():
    pos=torch.tensor([1.,1.,1.]);neg=torch.stack([torch.ones(150),torch.zeros(150),torch.full((150,),2.)])
    np.testing.assert_array_equal(official_ranks(pos,neg).numpy(),[76.,1.,151.])
    assert metrics([76,1,151])['mrr']==np.mean([1/76,1,1/151])


def test_matches_pinned_official_evaluator():
    from came_repro.task_metrics import ranking_metrics
    generator=torch.Generator().manual_seed(5)
    pos=torch.randint(-2,3,(113,),generator=generator).float()
    neg=torch.randint(-2,3,(113,150),generator=generator).float()
    ranks=official_ranks(pos,neg)
    torch.testing.assert_close(1/ranks.double(),ranking_metrics(pos,neg)['mrr'],rtol=0,atol=0)


def test_filtering_cannot_lower_mrr_for_fixed_predictions():
    pos=torch.tensor([.2,.8]);neg=torch.tensor([[.1,.5,.2],[.8,.9,.1]])
    original=official_ranks(pos,neg)
    filtered=neg.clone();filtered[:,1]=-torch.inf
    # Count comparison without letting nonfinite model scores into real evaluator.
    after=1.+.5*((filtered>=pos[:,None]).sum(1)+(filtered>pos[:,None]).sum(1))
    assert torch.all(after<=original) and torch.all(1/after>=1/original)


@pytest.mark.parametrize('values',[[],[0],[152],[np.nan],[np.inf]])
def test_invalid_ranks_rejected(values):
    with pytest.raises(ValueError):metrics(values)


def test_nonfinite_scores_rejected():
    with pytest.raises(FloatingPointError):official_ranks(torch.tensor([1.]),torch.tensor([[float('nan')]]))


def test_upstream_three_linear_hadamard_shape_and_symmetry():
    torch.manual_seed(1);head=make_head(8);a=torch.randn(4,8);b=torch.randn(4,8)
    assert sum(isinstance(m,torch.nn.Linear) for m in head)==3
    assert pair_logits(head,a,b).shape==(4,)
    torch.testing.assert_close(pair_logits(head,a,b),pair_logits(head,b,a),rtol=0,atol=0)


def test_predeclared_budget_preserves_full_official_test():
    import json
    p=json.loads((ROOT/'configs/goodreads_official_rebuild_20261021.json').read_text())
    assert p['recipe']['positive_budget_per_round']==1000*512
    assert p['official_test_queries']==1710677 and p['fixed_test_negatives']==150
    assert p['test_score_admission_gate'] is False and p['seeds']==list(range(10))


def test_checkpoint_reservation_rejects_escape_and_overflow(tmp_path,monkeypatch):
    import json
    from came_repro import goodreads_storage as storage
    root=tmp_path/'cache';run=root/'20261021T000000Z_goodreads_fixture';run.mkdir(parents=True)
    snapshot=tmp_path/'reservation.json'
    snapshot.write_text(json.dumps(dict(checkpoint_root=str(root),baseline_external_cache_bytes=0,baseline_owned_bytes=0,
        max_extra_cache_bytes=100,cache_cap_bytes=150,reserve_free_bytes=0)))
    monkeypatch.setattr(storage,'RESERVATION',snapshot)
    with pytest.raises(ValueError):storage.guard(tmp_path/'outside.pt',1)
    with pytest.raises(RuntimeError):storage.guard(run/'last.pt',101)
    (run/'existing.pt').write_bytes(b'x'*50)
    assert storage.owned_checkpoint_bytes(root)==50
    storage.guard(run/'last.pt',49)
    with pytest.raises(RuntimeError):storage.guard(run/'last.pt',51)


@pytest.mark.parametrize('bad_row',[0,4,8])
def test_full_row_finite_check_never_skips_tail(bad_row):
    from came_repro.goodreads_official import require_finite_rows
    x=torch.ones(9,8);require_finite_rows(x,4);x[bad_row,-1]=float('nan')
    with pytest.raises(FloatingPointError):require_finite_rows(x,4)


def test_audited_resume_only_ignores_known_release_metadata():
    import copy
    from came_repro.goodreads_official import resume_configuration_matches
    old=dict(seed=0,evaluation=dict(release_sha256='9e8b21a27661a85f74c19d4e0312cb21230c2e6823206b5c4f2c1c541ba2293b',lr=.0005))
    new=copy.deepcopy(old);new['evaluation']['release_sha256']='new_frozen_release'
    assert resume_configuration_matches(old,new)
    new['evaluation']['lr']=.001;assert not resume_configuration_matches(old,new)
    old['evaluation']['release_sha256']='unapproved_old_code';new['evaluation']['lr']=.0005
    assert not resume_configuration_matches(old,new)
