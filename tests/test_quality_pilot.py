"""Selection leakage, small-round identity, cached CUDA scoring and held-out test."""
import copy
import json
import logging

import numpy as np
import pytest
import torch
from came_repro.author_loader import ROOT
from came_repro.quality_protocol import validate_profile, validation_gate, select_graph_policies, extension_signature, QUALITY_DATASETS
from came_repro.quality_probe import CachedProbeInput, remove_owned_cache
from came_repro.downstream import ProbeInput, score_probe, train_linear_probe
from came_repro.quality_pipeline import screen_jobs, remaining_first_seed_jobs, active_clock
from came_repro.quality_pretrain import ContinuationSchedule
from came_repro import subset_offline_train


def profile():
    return json.loads((ROOT/'configs/quality_pilot.json').read_text())


def test_small_profile_has_no_deadline_and_no_ele_fashion():
    p = profile()
    validate_profile(p)
    assert p['max_active_hours'] is None
    assert p['confirmation_seeds']==[0]
    assert 'ele-fashion' not in p['datasets']
    assert not p['run_remaining_first_seed_tasks']
    assert all(j.get('seed',0)==0 for j in remaining_first_seed_jobs('reference'))
    assert len(screen_jobs(p))==9 and len(remaining_first_seed_jobs('reference'))==6
    assert active_clock()>0


def test_selection_uses_validation_not_test_scores():
    p = profile()
    reference = {d:{'validation_accuracy':.70,'test_accuracy':.99} for d in QUALITY_DATASETS}
    candidate = {d:{'validation_accuracy':.72,'test_accuracy':.01} for d in QUALITY_DATASETS}
    a = validation_gate(reference,candidate,p)
    assert a['passed']
    for v in candidate.values():
        v['test_accuracy']=.999
    assert validation_gate(reference,candidate,p)==a
    candidate['books-nc']['validation_accuracy']=.65
    assert not validation_gate(reference,candidate,p)['passed']
    assert select_graph_policies(reference,candidate,.5)['books-nc']=='original'


def test_extension_cannot_change_seed_pools_or_architecture():
    original = json.loads((ROOT/'configs/subset_offline_24h.json').read_text())
    new = copy.deepcopy(original)
    new.update(name='quality_extra',deviations=['one additional round'])
    new['training']['epochs']=9
    assert extension_signature(new)==extension_signature(original)
    new['sampling']['seed_limits']['books-lp'] += 32
    assert extension_signature(new)!=extension_signature(original)


def test_completed_parent_creates_ninth_order_once_and_partial_resume_keeps_cursor():
    counts={'a':20,'b':30}
    limits={'a':8,'b':12}
    parent=subset_offline_train.FixedSubsetSchedule(counts,4,torch.Generator().manual_seed(42),limits)
    parent.epoch=8
    parent.cursors={name:len(order) for name,order in parent.orders.items()}
    state=parent.state_dict()
    expected=subset_offline_train.FixedSubsetSchedule(counts,4,torch.Generator().manual_seed(42),limits)
    expected.load_state_dict(state)
    expected.new_epoch()
    extended=ContinuationSchedule(counts,4,torch.Generator().manual_seed(42),limits)
    extended.load_state_dict(state)
    assert extended.epoch==8 and extended.cursors=={'a':0,'b':0}
    for name in counts:
        torch.testing.assert_close(extended.orders[name],expected.orders[name],rtol=0,atol=0)
    assert extended.next_batch() is not None
    middle=extended.state_dict()
    resumed=ContinuationSchedule(counts,4,torch.Generator().manual_seed(42),limits)
    resumed.load_state_dict(middle)
    assert resumed.cursors==extended.cursors
    for name in counts:
        torch.testing.assert_close(resumed.orders[name],extended.orders[name],rtol=0,atol=0)
    subset_offline_train.CURRENT_SCHEDULE=None


def test_owned_cleanup_refuses_raw_or_entire_cache_root(tmp_path):
    storage={'embedding_cache_root':str(tmp_path/'embeddings')}
    for path in (tmp_path/'raw',tmp_path/'embeddings',tmp_path/'embeddings/quality_pilot'):
        path.mkdir(parents=True,exist_ok=True)
        with pytest.raises(ValueError):
            remove_owned_cache(path,storage)
    owned=tmp_path/'embeddings/quality_pilot/hash/wiki-cs_original'
    owned.mkdir(parents=True)
    (owned/'fixture').write_text('derived')
    remove_owned_cache(owned,storage)
    assert not owned.exists()
    assert (tmp_path/'raw').exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA required')
def test_cached_cuda_validation_matches_original_values_and_scores():
    rng=np.random.default_rng(42)
    features=rng.normal(size=(30,8)).astype(np.float32)
    ids=np.arange(1,25)
    labels=ids%3
    original=ProbeInput(features,labels,indices=ids)
    cached=CachedProbeInput(features,labels,indices=ids)
    cached.cache_validation(1,logging.getLogger('fixture'))
    a,b=original.batch(np.arange(7))
    c,d=cached.batch(np.arange(7))
    torch.testing.assert_close(a,c,rtol=0,atol=0)
    torch.testing.assert_close(b,d,rtol=0,atol=0)
    classifier=torch.nn.Linear(8,3).cuda()
    result_a=score_probe(classifier,original,batch_size=7,save_predictions=True)
    result_b=score_probe(classifier,cached,batch_size=7,save_predictions=True)
    assert result_a[:2]==pytest.approx(result_b[:2],abs=1e-7)
    np.testing.assert_array_equal(result_a[2],result_b[2])


class DiagnosticExperiment:
    def __init__(self,directory):
        self.directory=directory
        self.checkpoint_directory=directory/'checkpoints'
        for p in ('checkpoints','figures','logs'):
            (directory/p).mkdir(parents=True)
        self.config={'seed':0,'evaluation':{'dataset':'quality_cuda_fixture','epochs':3,'resume_head':None}}
        self.data_audit={'fixture':'held_out_quality_test'}
        self.logger=logging.getLogger('quality_fixture')
        (directory/'config.json').write_text(json.dumps(self.config))
    def metric(self,row):
        with (self.directory/'logs/metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA required')
def test_validation_only_training_never_accesses_test(tmp_path):
    class ForbiddenTest:
        def __getattribute__(self,name):
            raise AssertionError('Diagnostic touched held-out test: '+name)
    rng=np.random.default_rng(4)
    features=rng.normal(size=(12,4)).astype(np.float32)
    train=ProbeInput(features,np.arange(8)%2,indices=np.arange(8))
    valid=ProbeInput(features,np.arange(4)%2,indices=np.arange(8,12))
    experiment=DiagnosticExperiment(tmp_path/'experiment')
    result=train_linear_probe(train,valid,ForbiddenTest(),2,3,100,0,experiment,evaluate_test=False)
    assert result['test_evaluations']==0 and result['test_accuracy'] is None
    assert not (experiment.directory/'test_predictions.npz').exists()
    with np.load(experiment.directory/'selection_predictions.npz') as output:
        assert np.mean(output['prediction']==output['labels'])==result['validation_accuracy']
