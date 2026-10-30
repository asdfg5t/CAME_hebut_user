import sqlite3
from types import SimpleNamespace

import dgl
import numpy as np
import pytest
import torch

from came_repro import subset_offline_train as subset
from came_repro.ide_workflow import read_json


def test_subset_rounds_cover_same_fixed_unlabeled_pool_and_resume_next_order(monkeypatch):
    monkeypatch.setattr(subset, 'CURRENT_EXPERIMENT', None)
    counts={'a':21,'b':17}
    limits={'a':8,'b':5}
    schedule=subset.FixedSubsetSchedule(counts,3,torch.Generator().manual_seed(42),limits)
    pools={name:set(values.tolist()) for name,values in schedule.pools.items()}
    for round_id in range(3):
        seen={name:[] for name in counts}
        while True:
            batch=schedule.next_batch()
            if batch is None:
                break
            name,seeds=batch
            seen[name]+=seeds.tolist()
        assert all(set(seen[name])==pools[name] and len(seen[name])==limits[name] for name in counts)
        if round_id==0:
            state=schedule.state_dict()
            clone=subset.FixedSubsetSchedule(counts,3,torch.Generator().manual_seed(42),limits)
            clone.load_state_dict(state)
            schedule.new_epoch()
            clone.new_epoch()
            for name in counts:
                torch.testing.assert_close(schedule.orders[name],clone.orders[name],rtol=0,atol=0)
        elif round_id<2:
            schedule.new_epoch()


def test_offline_ppr_computes_before_training_and_rejects_unprepared_seed(tmp_path,monkeypatch):
    import came_repro.joint_sampling as packed
    monkeypatch.setattr(packed,'ROOT',tmp_path)
    monkeypatch.setattr(subset,'CURRENT_EXPERIMENT',None)
    monkeypatch.setattr(subset,'CURRENT_SCHEDULE',SimpleNamespace(pools={'fixture':torch.tensor([0,1])}))
    graph=dgl.graph(([0,1,2],[1,2,0]),num_nodes=4)
    data=SimpleNamespace(graph=graph,audit={'dataset':'fixture','view_files_sha256':{'edge_index.npy':'cycle_fixture'}})
    config={'direction':'official_directed','top_k':3,'alpha':.15,'epsilon':1e-5,'max_pushes':100000}
    sampler=subset.OfflinePPRSampler(data,config)
    try:
        assert sampler.strict_offline
        assert sampler.connection.execute('SELECT COUNT(*) FROM ppr').fetchone()[0]==2
        nodes,_=sampler.neighbors(0)
        assert 0 in nodes and nodes.dtype==np.int64
        with pytest.raises(RuntimeError,match='online expansion is forbidden'):
            sampler.neighbors(3)
        assert sampler.connection.execute('SELECT COUNT(*) FROM ppr').fetchone()[0]==2
    finally:
        sampler.close()


def test_subset_checkpoint_and_summary_never_claim_full_graph_epochs(tmp_path):
    path=tmp_path/'subset.pt'
    subset.save_subset_checkpoint(path,{'schema':'came_joint_v1','schedule':{'epoch':3},'model':{'fixture':torch.ones(2)}})
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['completed_epochs']==0
    assert saved['completed_subset_rounds']==3 and saved['schedule']['epoch']==3
    summary=tmp_path/'summary.json'
    subset.write_subset_json(summary,{'seed_coverage_current_epoch':{'a':8},'completed_epochs':8,'complete_declared_pretraining':True})
    report=read_json(summary)
    assert report['completed_epochs']==0 and report['completed_subset_rounds']==8
    assert not report['complete_declared_pretraining'] and report['complete_declared_subset_pretraining']
