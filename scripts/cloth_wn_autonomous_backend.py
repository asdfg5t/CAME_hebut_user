"""Reuse frozen CUDA trainers for a predeclared, validation-only LR grid."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
PROFILE = ROOT/'configs/cloth_wn_autonomous_20261017.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def candidate_by_id(name):
    plan=read(PROFILE)
    for task in ('cloth','wn'):
        for candidate in plan[task+'_candidates']:
            if candidate['id']==name:return task,candidate
    raise ValueError('Undeclared candidate: '+name)


def configured_probe(candidate,*args,**kwargs):
    from came_repro.downstream import train_linear_probe
    kwargs['lr']=candidate['lr']
    return train_linear_probe(*args,**kwargs)


def cloth_probe(candidate,checkpoint,resume):
    import torch
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.experiment import Experiment
    from came_repro.protocol_repair import candidate_cache,check_cuda
    from came_repro.lp_validation_repair import load_verified_cache
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro.pause_control import TrainingPaused
    from run_lp_hard_negative_candidate import train_hard
    check_cuda();torch.set_num_threads(4)
    storage=read(ROOT/'configs/storage.json');digest=fingerprint(checkpoint)
    cache=candidate_cache('cloth-copurchase',digest,storage)
    gpu,audit=load_verified_cache('cloth-copurchase',cache,digest)
    if audit.get('diagnostic_topology',{}).get('policy')!='upstream_train_bidirected_unique':
        raise ValueError('Full official train-only topology required')
    recipe=copy.deepcopy(read(ROOT/'configs/representation_coverage_validation_20261016.json')['cloth_recipe'])
    recipe.update({k:candidate[k] for k in ('lr','max_epochs','patience')})
    config=copy.deepcopy(torch.load(str(checkpoint),map_location='cpu',weights_only=False)['config'])
    config.update(name='auto_'+candidate['id'],seed=0,storage=storage)
    config['evaluation']=dict(recipe,dataset='cloth-copurchase',task='link',seed=0,
        encoder_sha256=digest,embedding_cache=str(cache),negative_pool_size=4,
        autonomous_candidate=candidate['id'],autonomous_profile_sha256=fingerprint(PROFILE),test_used_for_selection=False)
    experiment=Experiment(config,audit,resume or checkpoint)
    try:
        raw=ROOT/'data/raw/mm_graph/cloth-copurchase/lp-edge-split.pt'
        if fingerprint(raw)!=audit['raw_sha256']['lp-edge-split.pt']:raise ValueError('Official split changed')
        split=torch.load(str(raw),map_location='cpu',weights_only=False,mmap=True)
        with use_cached_vectors():summary=train_hard(gpu,split['train'],split['valid'],recipe,0,experiment,resume)
        summary.update(dataset='cloth-copurchase',task='link',representation='came',encoder_sha256=digest,
            embedding_cache=str(cache),graph_policy='upstream_train_bidirected_unique',
            autonomous_candidate=candidate['id'],diagnostic_only=True,paper_reproduction_complete=False)
        atomic_write_json(experiment.directory/'summary.json',summary)
        experiment.status('completed_autonomous_cloth_validation_only')
    except BaseException as error:
        experiment.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
    finally:experiment.close()


def wn_probe(candidate,checkpoint,resume):
    from came_repro import kg_affine_probe as kg
    from came_repro.author_loader import fingerprint
    from came_repro.protocol_repair import candidate_cache
    cache=candidate_cache('WN18RR',fingerprint(checkpoint),read(ROOT/'configs/storage.json'))
    original_experiment=kg.Experiment
    original_trainer=kg.train_linear_probe
    def experiment_factory(config,audit,parent):
        config=copy.deepcopy(config);config['name']='auto_'+candidate['id']
        config['evaluation'].update(autonomous_candidate=candidate['id'],optimizer_lr=candidate['lr'],
            autonomous_profile_sha256=fingerprint(PROFILE),test_used_for_selection=False)
        return original_experiment(config,audit,parent)
    def train(*args,**kwargs):return configured_probe(candidate,*args,**kwargs)
    kg.Experiment=experiment_factory;kg.train_linear_probe=train
    previous=sys.argv
    sys.argv=['kg_affine_probe','--dataset','WN18RR','--checkpoint',str(checkpoint),
              '--embedding-cache',str(cache),'--graph-policy','train_reverse','--seed','0']
    if resume:sys.argv+=['--resume-head',str(resume)]
    try:kg.main()
    finally:kg.Experiment=original_experiment;kg.train_linear_probe=original_trainer;sys.argv=previous


def assert_same(left,right):
    import numpy as np
    import torch
    if isinstance(left,torch.Tensor):
        if not torch.equal(left.cpu(),right.cpu()):raise ValueError('Actual CUDA resume tensor mismatch')
    elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        if left.keys()!=right.keys():raise ValueError('Resume keys differ')
        for key in left:assert_same(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right):raise ValueError('Resume length differs')
        for a,b in zip(left,right):assert_same(a,b)
    elif left!=right:raise ValueError('Actual CUDA resume scalar mismatch')


def acceptance():
    import numpy as np
    import torch
    from came_repro.protocol_repair import check_cuda
    from came_repro.experiment import Experiment,seed_everything
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro.downstream import ProbeInput
    import came_repro.downstream as downstream
    from came_repro.pause_control import TrainingPaused
    import came_repro.pause_control as pauses
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    from run_lp_hard_negative_candidate import train_hard
    check_cuda();torch.set_num_threads(4);seed_everything(713)
    gpu=torch.randn(16,1024,device='cuda')
    matrix=gpu.cpu().numpy()
    pairs=np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6]],dtype=np.int64)
    valid_pairs=np.array([[8,9],[9,10]],dtype=np.int64)
    labels=np.array([0,1,2,0,1,2]);vlabels=np.array([0,1])
    cases=[]
    for task in ('cloth','wn'):
        recipe=dict(head_variant='two_hidden_layernorm_head_upstream_schedule',lr=.0005,batch_size=3,
                    max_epochs=3,patience=10,step_lr_size=None,step_lr_gamma=None,loss_reduction='separate_mean_sum')
        candidate={'lr':.01}
        config={'name':'auto_'+task+'_resume_fixture','seed':17,'storage':read(ROOT/'configs/storage.json'),
                'evaluation':{'fixture_only':True,'dataset':task,'optimizer_lr':recipe['lr'] if task=='cloth' else .01}}
        audit={'fixture_only':True,'not_a_benchmark_result':True}
        train={'source_node':pairs[:,0],'target_node':pairs[:,1]}
        valid={'source_node':valid_pairs[:,0],'target_node':valid_pairs[:,1],
               'target_node_neg':np.tile(np.arange(16),(2,10))[:,:150]}
        def execute(experiment,resume=None):
            if task=='cloth':
                with use_cached_vectors():return train_hard(gpu,train,valid,recipe,17,experiment,resume)
            return configured_probe(candidate,ProbeInput(matrix,labels,pairs=pairs),
                ProbeInput(matrix,vlabels,pairs=valid_pairs),None,3,3,10,17,experiment,
                resume=resume,evaluate_test=False)
        baseline=Experiment(config,audit)
        try:execute(baseline);baseline.status('completed_autonomous_fixture_reference')
        finally:baseline.close()
        original=pauses.pause_requested;original_probe_pause=downstream.pause_requested;calls=[0]
        def pause_after_two():calls[0]+=1;return calls[0]>=2
        pauses.pause_requested=pause_after_two
        downstream.pause_requested=pause_after_two
        paused=Experiment(config,audit)
        try:
            try:execute(paused)
            except TrainingPaused:paused.status('paused_autonomous_fixture')
            else:raise ValueError('Resume fixture did not pause')
        finally:
            paused.close();pauses.pause_requested=original;downstream.pause_requested=original_probe_pause
        name='link_head_last.pt' if task=='cloth' else 'probe_last.pt'
        cp=paused.checkpoint_directory/name
        resumed=Experiment(config,audit,cp)
        try:execute(resumed,cp);resumed.status('completed_autonomous_fixture_resume')
        finally:resumed.close()
        a=torch.load(str(baseline.checkpoint_directory/name),map_location='cpu',weights_only=False)
        b=torch.load(str(resumed.checkpoint_directory/name),map_location='cpu',weights_only=False)
        for key in ('optimizer','rng','sampling_generator','history_rows',
                    'head' if task=='cloth' else 'classifier','best_state','selected_epoch','next_epoch'):
            assert_same(a[key],b[key])
        cases.append({'task':task,'passed':True,'seed':17,'model_Adam_RNG_sampling_history_bitexact':True,
            'fixture_only':True,'reference_run':str(baseline.directory),'paused_run':str(paused.directory),
            'resumed_run':str(resumed.directory),'parent_sha256':fingerprint(cp)})
    output=ROOT/'artifacts/cloth_wn_autonomous_20261017/acceptance.json'
    atomic_write_json(output,{'passed':True,'actual_Torch_and_DGL_CUDA':True,'cases':cases,
        'backend_sha256':fingerprint(__file__),'profile_sha256':fingerprint(PROFILE),
        'fixture_only_not_benchmark':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['acceptance','probe'],required=True)
    p.add_argument('--candidate');p.add_argument('--checkpoint',type=Path);p.add_argument('--resume-head',type=Path)
    a=p.parse_args()
    if a.stage=='acceptance':acceptance();return
    task,candidate=candidate_by_id(a.candidate)
    if a.checkpoint is None:p.error('A complete frozen encoder checkpoint is required')
    if task=='cloth':cloth_probe(candidate,a.checkpoint,a.resume_head)
    else:wn_probe(candidate,a.checkpoint,a.resume_head)


if __name__=='__main__':main()
