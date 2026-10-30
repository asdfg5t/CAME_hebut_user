"""Actual CUDA head updates: uninterrupted and resumed epochs must agree."""
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from came_repro.downstream import ProbeInput, train_linear_probe
from came_repro.link_evaluation import train_link_head


class TestExperiment:
    __test__ = False

    def __init__(self,directory,epochs):
        self.directory=Path(directory)
        self.checkpoint_directory=self.directory/'checkpoints'
        for name in ('checkpoints','figures','logs'):
            (self.directory/name).mkdir(parents=True)
        self.config={'seed':7,'evaluation':{'dataset':'cuda_fixture','epochs':epochs,'resume_head':None}}
        self.data_audit={'fixture':'eight_node_cuda_resume_v1'}
        self.logger=logging.getLogger(str(directory))
        (self.directory/'config.json').write_text(json.dumps(self.config))

    def metric(self,row):
        with (self.directory/'logs/metrics.jsonl').open('a') as output:
            output.write(json.dumps(row)+'\n')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA required')
@pytest.mark.parametrize('task',['linear','link'])
def test_head_resume_matches_continuous_next_epoch(tmp_path,task):
    features=np.random.default_rng(8).normal(size=(8,4)).astype(np.float32)
    continuous,first,resumed=[TestExperiment(tmp_path/name,epochs) for name,epochs in (('continuous',2),('first',1),('resumed',2))]
    if task=='linear':
        labels=np.arange(8)%2
        inputs=[ProbeInput(features,labels[index],indices=index) for index in (np.arange(4),np.array([4,5]),np.array([6,7]))]
        def run(experiment,epochs,resume=None):
            return train_linear_probe(*inputs,2,epochs,5,7,experiment,batch_size=2,resume=resume)
        filename,key='probe_last.pt','classifier'
    else:
        split={'train':{'source_node':np.array([0,1,2,3]),'target_node':np.array([1,2,3,4])},
               'valid':{'source_node':np.array([0,2]),'target_node':np.array([2,4]),'target_node_neg':np.array([[3,4,5],[0,5,6]])},
               'test':{'source_node':np.array([5,6]),'target_node':np.array([6,7]),'target_node_neg':np.array([[1,2,3],[0,1,2]])}}
        def run(experiment,epochs,resume=None):
            return train_link_head(features,split,epochs,5,7,experiment,batch_size=2,eval_batch_size=1,resume=resume)
        filename,key='link_head_last.pt','head'
    run(continuous,2)
    run(first,1)
    result=run(resumed,2,first.checkpoint_directory/filename)
    a=torch.load(continuous.checkpoint_directory/filename,map_location='cpu',weights_only=False)
    b=torch.load(resumed.checkpoint_directory/filename,map_location='cpu',weights_only=False)
    assert a['completed_epochs']==b['completed_epochs']==2
    assert b['next_epoch']==3
    assert len(b['history_rows'])==2
    assert result['test_evaluations']==1
    for name in a[key]:
        torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
    assert b['sampling_generator']==a['sampling_generator']
    if task=='link':
        with np.load(resumed.directory/'test_link_ranks.npz') as values:
            assert float(np.mean(1/values['rank']))==pytest.approx(result['test']['official']['mrr'])
            assert int(values['reverse_overlap'].sum())==result['test']['reverse_overlap_queries']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA required')
@pytest.mark.parametrize('task',['linear','link'])
@pytest.mark.parametrize('pause_epoch',[1,2])
def test_safe_pause_can_resume_last_epoch_without_retraining_or_changing_selection(tmp_path,monkeypatch,task,pause_epoch):
    from came_repro.pause_control import TrainingPaused
    from came_repro import downstream,link_evaluation
    features=np.random.default_rng(8).normal(size=(8,4)).astype(np.float32)
    continuous,paused,resumed=[TestExperiment(tmp_path/name,2) for name in ('continuous','paused','resumed')]
    if task=='linear':
        labels=np.arange(8)%2
        inputs=[ProbeInput(features,labels[index],indices=index) for index in (np.arange(4),np.array([4,5]),np.array([6,7]))]
        def run(experiment,resume=None):
            return train_linear_probe(*inputs,2,2,5,7,experiment,batch_size=2,resume=resume)
        module,filename,key=downstream,'probe_last.pt','classifier'
    else:
        split={'train':{'source_node':np.array([0,1,2,3]),'target_node':np.array([1,2,3,4])},
               'valid':{'source_node':np.array([0,2]),'target_node':np.array([2,4]),'target_node_neg':np.array([[3,4,5],[0,5,6]])},
               'test':{'source_node':np.array([5,6]),'target_node':np.array([6,7]),'target_node_neg':np.array([[1,2,3],[0,1,2]])}}
        def run(experiment,resume=None):
            return train_link_head(features,split,2,5,7,experiment,batch_size=2,eval_batch_size=1,resume=resume)
        module,filename,key=link_evaluation,'link_head_last.pt','head'
    run(continuous)
    original_metric=paused.metric
    recorded=[]
    def metric(row):
        original_metric(row)
        recorded.append(row['step'])
    paused.metric=metric
    monkeypatch.setattr(module,'pause_requested',lambda:bool(recorded) and recorded[-1]>=pause_epoch)
    with pytest.raises(TrainingPaused):
        run(paused)
    assert not list(paused.directory.glob('test*.npz'))
    state=torch.load(paused.checkpoint_directory/filename,map_location='cpu',weights_only=False)
    assert state['completed_epochs']==pause_epoch
    monkeypatch.setattr(module,'pause_requested',lambda:False)
    result=run(resumed,paused.checkpoint_directory/filename)
    assert result['test_evaluations']==1 and result['epochs_executed']==2
    a=torch.load(continuous.checkpoint_directory/filename,map_location='cpu',weights_only=False)
    if pause_epoch==1:
        b=torch.load(resumed.checkpoint_directory/filename,map_location='cpu',weights_only=False)
    else:
        b=state
    for name in a[key]:
        torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
