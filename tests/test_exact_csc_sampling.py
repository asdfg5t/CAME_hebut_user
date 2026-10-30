import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import dgl

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from came_repro.exact_csc_sampling import ExactCSCGraph,ExactCSCSampler
from came_repro.compact_csc import build_index
from run_12_products_csc64_repair import validate_sampled_blocks


@pytest.mark.parametrize('targets',[[5,0],[2],[0,1,2,3,4,5]])
def test_complete_neighbor_multisets_match_dgl_for_every_layer(tmp_path,targets):
    edges=np.array([[0,1],[0,1],[1,2],[2,0],[3,3],[3,3],[4,2]])
    build_index(edges,6,tmp_path,'fixture')
    graph=ExactCSCGraph(tmp_path);sampler=ExactCSCSampler(2)
    seeds=torch.tensor(targets);inputs,outputs,blocks=sampler.sample_blocks(graph,seeds)
    validate_sampled_blocks(blocks,inputs)
    assert torch.equal(outputs,seeds)
    raw=dgl.add_self_loop(dgl.remove_self_loop(dgl.graph((edges[:,0],edges[:,1]),num_nodes=6)))
    _,_,reference=dgl.dataloading.MultiLayerFullNeighborSampler(2).sample_blocks(raw,seeds)
    def edges_global(block):
        a,b=block.edges(order='eid')
        keys=block.srcdata[dgl.NID][a]*6+block.dstdata[dgl.NID][b]
        return keys.sort().values
    for candidate,expected in zip(blocks,reference):
        assert torch.equal(edges_global(candidate),edges_global(expected))
        assert torch.equal(candidate.dstdata[dgl.NID].sort().values,expected.dstdata[dgl.NID].sort().values)
    assert graph.num_edges()==11 and graph.num_nodes()==6


def test_reject_modified_index_and_invalid_targets(tmp_path):
    build_index(np.array([[0,1]]),3,tmp_path,'fixture')
    graph=ExactCSCGraph(tmp_path)
    with pytest.raises(ValueError):graph.block([1,1])
    with pytest.raises(ValueError):graph.block([-1])
    with pytest.raises(ValueError):graph.block([3])
    with pytest.raises(ValueError):ExactCSCSampler(1)
    graph.indptr._mmap.close();graph.indices._mmap.close();del graph
    values=np.load(tmp_path/'indices.npy').copy();values[0]=2;np.save(tmp_path/'indices.npy',values)
    with pytest.raises(ValueError,match='changed'):ExactCSCGraph(tmp_path)
