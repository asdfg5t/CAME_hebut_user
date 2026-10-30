import importlib.util
from pathlib import Path
import numpy as np
import torch

path=Path(__file__).resolve().parents[1]/'scripts/run_lp_hard_negative_candidate.py'
spec=importlib.util.spec_from_file_location('hard_negative_candidate',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_mining_uses_only_known_train_nonedges_and_correct_max(monkeypatch):
    from came_repro import link_evaluation as link
    from came_repro.lp_validation_repair import TrainOnlyNegatives
    n=12;features=torch.arange(n,dtype=torch.float32).reshape(n,1).repeat(1,4)
    source=np.array([0,1,2,3]);target=source+1
    sampler=TrainOnlyNegatives(np.concatenate((source,target)),np.concatenate((target,source)),n)
    class Head(torch.nn.Module):
        def forward(self,a,b):return b.sum(-1)
    head=Head().train()
    monkeypatch.setattr(link,'vectors',lambda x,ids:x[torch.as_tensor(ids,dtype=torch.long)])
    chosen,pool,scores=module.sample_hard_negative(head,features,source,sampler,np.random.default_rng(0))
    np.testing.assert_array_equal(chosen,pool.max(1))
    keys=np.unique(np.concatenate((source*n+target,target*n+source)))
    assert not np.isin(np.repeat(source,4)*n+pool.ravel(),keys).any()
    assert not (pool==source[:,None]).any() and head.training
    assert not scores.requires_grad


def test_fixed_training_generator_reproduces_same_pool_and_selection(monkeypatch):
    from came_repro import link_evaluation as link
    from came_repro.lp_validation_repair import TrainOnlyNegatives
    features=torch.randn(12,4);source=np.array([0,1,2]);target=source+1
    sampler=TrainOnlyNegatives(np.concatenate((source,target)),np.concatenate((target,source)),12)
    class Head(torch.nn.Module):
        def forward(self,a,b):return (a*b).sum(-1)
    head=Head().eval();monkeypatch.setattr(link,'vectors',lambda x,ids:x[torch.as_tensor(ids,dtype=torch.long)])
    a=module.sample_hard_negative(head,features,source,sampler,np.random.default_rng(17));b=module.sample_hard_negative(head,features,source,sampler,np.random.default_rng(17))
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);assert torch.equal(a[2],b[2]) and not head.training
