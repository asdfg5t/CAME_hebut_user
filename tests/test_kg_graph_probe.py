import numpy as np
import pytest
import torch
import dgl
from came_repro.kg_graph_probe import training_reverse_graph,choose_graph_policy,validate_confirmation_lock


def test_reverse_view_only_contains_train_edges_and_their_reverse_with_parallel_edges():
    g=dgl.graph(([0,0,1],[1,1,2]),num_nodes=4)
    transformed=training_reverse_graph(g)
    assert transformed.num_nodes()==4 and transformed.num_edges()==6
    source,target=transformed.edges()
    assert list(zip(source.tolist(),target.tolist())).count((0,1))==2
    assert list(zip(source.tolist(),target.tolist())).count((1,0))==2
    assert set(zip(source.tolist(),target.tolist()))=={(0,1),(1,0),(1,2),(2,1)}


def test_policy_gate_uses_validation_only_and_keeps_paper_linear_head():
    lock={'dataset':'WN18RR','encoder_sha256':'same','policy':'train_reverse','locked_before_test':True,
          'test_used_for_selection':False,'reference_validation_accuracy':.77,'candidate_validation_accuracy':.81,
          'edge_representation':'head_tail_concatenation','classifier':'linear_cross_entropy_lr_0.05_max2000_patience100'}
    validate_confirmation_lock(lock,'WN18RR','same','train_reverse')
    for key,value in [('test_used_for_selection',True),('edge_representation','hadamard'),('encoder_sha256','different')]:
        bad=dict(lock);bad[key]=value
        with pytest.raises(ValueError):validate_confirmation_lock(bad,'WN18RR','same','train_reverse')
    assert choose_graph_policy(.8,.801)=='original'
    with pytest.raises(ValueError):choose_graph_policy(.8,float('nan'))
