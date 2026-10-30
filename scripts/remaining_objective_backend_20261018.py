"""One listwise Cloth candidate and one normalized-linear WN solver candidate."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
PROFILE = ROOT/'configs/remaining_objectives_20261018.json'
OUT = ROOT/'artifacts/remaining_objectives_20261018'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_plan():
    from came_repro.author_loader import fingerprint, verify_originals
    verify_originals(); plan = read(PROFILE)
    for path, digest in plan['dependencies'].items():
        if fingerprint(ROOT/path) != digest: raise ValueError('Frozen objective dependency changed: '+path)
    if plan['seed'] != 0 or plan['minimum_validation_gain_pp'] != .5 or plan['maximum_tests_per_task'] != 1:
        raise ValueError('Fixed validation-only finite plan required')
    return plan


def acceptance():
    import torch
    import numpy as np
    from came_repro.experiment import Experiment, seed_everything
    from came_repro.listwise_link_train import train_listwise
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro import pause_control as pauses
    from cloth_wn_autonomous_backend import assert_same
    from run_convergence_candidate import train_wolfe
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    import unresolved_backend_20261018 as base
    seed_everything(713); gpu = torch.randn(16,1024,device='cuda')
    train = dict(source_node=np.arange(6), target_node=np.arange(6)+1)
    valid = dict(source_node=np.array([8,9]),target_node=np.array([9,10]),target_node_neg=np.tile(np.arange(16),(2,10))[:,:150])
    recipe = copy.deepcopy(base.read(base.PROFILE)['cloth']['recipe']); recipe.update(batch_size=3,max_epochs=3)
    x = torch.randn(32,8,device='cuda'); y = (x[:,0]>0).long()
    class Fixture:
        width=8
        def __len__(self): return len(y)
        def batch(self,index): return x[index].clone(),y[index].clone()
    fixture = Fixture(); runs=[]
    for task in ('cloth','wn'):
        config=dict(name='objectives_'+task+'_resume_fixture',seed=0,storage=base.storage(),evaluation=dict(fixture_only=True,profile_sha256=fingerprint(PROFILE)))
        def train_one(experiment,parent=None):
            if task=='cloth':
                with use_cached_vectors(): return train_listwise(gpu,train,valid,recipe,0,experiment,parent)
            return train_wolfe(fixture,fixture,2,dict(lr=.05,max_epochs=3,patience=100),experiment,parent)
        reference=Experiment(config,dict(fixture_only_not_paper_dataset=True))
        try: train_one(reference);reference.status('completed_fixture')
        finally: reference.close()
        original=pauses.pause_requested;calls=[0]
        def pause_two(): calls[0]+=1;return calls[0]>=2
        paused=Experiment(config,dict(fixture_only_not_paper_dataset=True));pauses.pause_requested=pause_two
        try:
            try: train_one(paused)
            except pauses.TrainingPaused: paused.status('paused_fixture')
            else: raise ValueError('Fixture pause did not trigger')
        finally: pauses.pause_requested=original;paused.close()
        name='link_head_last.pt' if task=='cloth' else 'probe_last.pt'
        parent=paused.checkpoint_directory/name
        resumed=Experiment(config,dict(fixture_only_not_paper_dataset=True),parent)
        try: train_one(resumed,parent);resumed.status('completed_fixture_resume')
        finally: resumed.close()
        a=torch.load(str(reference.checkpoint_directory/name),map_location='cpu',weights_only=False)
        b=torch.load(str(resumed.checkpoint_directory/name),map_location='cpu',weights_only=False)
        keys=['head' if task=='cloth' else 'classifier','optimizer','rng','history_rows','best_state','selected_epoch','next_epoch']
        if task=='cloth':keys+=['sampling_generator']
        for key in keys:assert_same(a[key],b[key])
        if a['next_epoch']!=4:raise ValueError('Incomplete fixture history')
        runs.extend(str(e.directory) for e in (reference,paused,resumed))
    atomic_write_json(OUT/'acceptance.json',dict(passed=True,actual_Torch_DGL_CUDA=True,
        full_head_Adam_or_LBFGS_RNG_history_resume_verified=True,fixture_not_benchmark=True,
        fixture_runs=runs,profile_sha256=fingerprint(PROFILE),backend_sha256=fingerprint(__file__)))


def stable_linear(train,valid,test,classes,epochs,patience,seed,experiment,**kwargs):
    import numpy as np
    import torch
    from run_convergence_candidate import train_wolfe
    from came_repro.experiment import atomic_checkpoint
    if test is not None or kwargs.get('evaluate_test',False):raise ValueError('Validation-only solver required')
    recipe=read(PROFILE)['wn_solver']
    summary,pred,best,history=train_wolfe(train,valid,classes,recipe,experiment,kwargs.get('resume'))
    atomic_checkpoint(experiment.checkpoint_directory/'probe_best.pt',dict(classifier=best,selected_epoch=summary['selected_epoch']))
    path=experiment.checkpoint_directory/'probe_last.pt'
    state=torch.load(str(path),map_location='cpu',weights_only=False)
    state['sampling_generator']=dict(kind='none_deterministic_full_batch')
    atomic_checkpoint(path,state)
    summary.update(seed=seed,optimization_steps=len(history),full_train_gradient_evaluations=sum(r['closure_evaluations'] for r in history),
        step_semantics='outer optimizer updates; multiple full-train objective/gradient passes per update; not paper epochs',
        selection_criterion='accuracy')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));steps=[r['step'] for r in history]
    for key,label in [('train_loss','train'),('val_loss','validation')]:axes[0].plot(steps,[r[key] for r in history],label=label)
    axes[0].legend();axes[0].set(xlabel='Optimizer update',ylabel='Cross entropy')
    axes[1].plot(steps,[r['val_accuracy'] for r in history]);axes[1].axvline(summary['selected_epoch'],ls='--',color='black')
    axes[1].set(xlabel='Optimizer update',ylabel='Official validation accuracy')
    fig.text(.02,.01,'WN seed0; fixed node-unit-L2 cache; linear CE/strong-Wolfe; full-train closures logged; test held out.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(experiment.directory/'figures/probe_curves.png',dpi=180);plt.close(fig)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['acceptance','cloth','wn'],required=True)
    parser.add_argument('--resume-head',type=Path);args=parser.parse_args();plan=verify_plan()
    import torch
    from came_repro.protocol_repair import check_cuda
    from came_repro.author_loader import fingerprint
    check_cuda();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='acceptance':acceptance();return
    receipt=read(OUT/'acceptance.json')
    if not receipt['passed'] or receipt['profile_sha256']!=fingerprint(PROFILE) or receipt['backend_sha256']!=fingerprint(__file__):
        raise ValueError('Matching actual CUDA/resume acceptance required')
    import unresolved_backend_20261018 as base
    from came_repro import experiment as experiments,cloth_regularized_train as cloth,kg_affine_probe as kg
    from came_repro.listwise_link_train import train_listwise
    old_factory,old_kg_factory,old_train,old_solver=experiments.Experiment,kg.Experiment,cloth.train_regularized,kg.train_linear_probe
    def factory(config,audit,parent=None):
        config=copy.deepcopy(config);config['name']='objective_'+args.stage+'_candidate'
        config['evaluation'].update(objective_profile_sha256=fingerprint(PROFILE))
        if args.stage=='cloth':config['evaluation'].update(loss_reduction='listwise_cross_entropy',training_candidates=5)
        else:config['evaluation'].update(optimizer='LBFGS_strong_wolfe',optimization_steps_cap=plan['wn_solver']['max_epochs'],node_unit_L2_fixed=True)
        config['deviations']=list(config.get('deviations',[]))+[plan[args.stage+'_deviation']]
        return old_factory(config,audit,parent)
    experiments.Experiment=factory;kg.Experiment=factory
    try:
        if args.stage=='cloth':cloth.train_regularized=train_listwise;base.cloth(args.resume_head)
        else:kg.train_linear_probe=stable_linear;base.wn(args.resume_head)
    finally:experiments.Experiment=old_factory;kg.Experiment=old_kg_factory;cloth.train_regularized=old_train;kg.train_linear_probe=old_solver


if __name__=='__main__':main()
