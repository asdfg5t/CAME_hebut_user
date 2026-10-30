import numpy as np
import pytest
import torch

from came_repro.task_metrics import ranking_metrics, support_query_episode, reverse_overlap_mask


def test_ranking_uses_average_rank_for_ties():
    positive=torch.tensor([.5,.5,.5])
    negative=torch.tensor([[.1,.2,.3],[.6,.5,.1],[.8,.7,.6]])
    result=ranking_metrics(positive,negative)
    torch.testing.assert_close(result['mrr'],torch.tensor([1.,1/2.5,1/4],dtype=torch.double))
    torch.testing.assert_close(result['hits@1'],torch.tensor([1.,0.,0.],dtype=torch.double))
    torch.testing.assert_close(result['hits@3'],torch.tensor([1.,1.,0.],dtype=torch.double))


def test_few_shot_support_queries_disjoint_and_repeatable():
    labels=np.repeat(np.arange(7),12)
    classes,support,query=support_query_episode(labels,np.arange(len(labels)),5,3,17)
    assert support.shape==(5,3)
    assert not set(support.flatten()) & set(query)
    assert len(query)==5*9
    for class_id,index in zip(classes,support):
        assert (labels[index]==class_id).all()
    again=support_query_episode(labels,np.arange(len(labels)),5,3,17)
    for left,right in zip((classes,support,query),again):
        np.testing.assert_array_equal(left,right)


def test_reverse_overlap_identifies_only_seen_node_pairs():
    mask=reverse_overlap_mask(np.array([0,1]),np.array([1,2]),np.array([1,2,3]),np.array([0,1,4]),5)
    np.testing.assert_array_equal(mask,[True,True,False])
