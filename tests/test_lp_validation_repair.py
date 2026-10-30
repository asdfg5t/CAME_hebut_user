import numpy as np
import pytest
import torch
from came_repro.lp_validation_repair import make_head,TrainOnlyNegatives
from came_repro.validation_repair_pipeline import choose_lp_candidate


def test_layernorm_head_is_rowwise_symmetric_and_preserves_full_width():
    torch.manual_seed(871)
    head=make_head('two_hidden_layernorm_head_upstream_schedule').eval()
    a,b=torch.randn(7,1024),torch.randn(7,1024)
    with torch.no_grad():
        full=head(a,b);split=torch.cat([head(a[:3],b[:3]),head(a[3:],b[3:])]);reverse=head(b,a)
    assert torch.allclose(full,split,rtol=1e-5,atol=1e-6)
    assert torch.equal(full,reverse)
    assert full.shape==(7,)
    assert head.network[0].in_features==head.network[0].out_features==1024


def test_sampler_excludes_only_train_edges_and_self_pairs():
    source=np.array([0,0,1,2]);target=np.array([1,2,2,3])
    sampler=TrainOnlyNegatives(source,target,7);random=np.random.default_rng(32)
    seeds=np.tile(np.arange(4),100)
    negative=sampler.sample(seeds,random)
    assert np.all(seeds!=negative)
    assert not np.isin(seeds*7+negative,sampler.keys).any()
    with pytest.raises(ValueError):make_head('arbitrary_test_tuned_head')


def test_lp_selection_reads_validation_only_with_declared_gain():
    reference={'summary':{'validation_mrr':.3,'test':{'official':{'mrr':1.}}}}
    small={'summary':{'validation_mrr':.304,'test':None}}
    better={'summary':{'validation_mrr':.32,'test':None}}
    assert choose_lp_candidate([small],reference) is reference
    assert choose_lp_candidate([small,better],reference) is better
