"""Independent bounded validation candidates; historical production sources stay frozen."""
import argparse
import copy
import hashlib
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def migrated_lp_state(old, config, budget):
    """Extend only a finished max-epoch budget, with explicit new identity and parent."""
    if old['schema'] != 'came_lp_protocol_v1' or old['next_epoch'] != old['completed_epochs']+1:
        raise ValueError('Complete epoch-boundary LP checkpoint required')
    before = old['config']['evaluation']
    if old['completed_epochs'] != before['max_epochs'] or old['stale'] >= before['patience']:
        raise ValueError('Only a budget-limited, non-early-stopped parent may be extended')
    if budget <= before['max_epochs']:
        raise ValueError('Budget must increase')
    expected = copy.deepcopy(old['config'])
    expected['name'] = config['name']
    expected['evaluation']['max_epochs'] = budget
    expected['evaluation']['convergence_profile_sha256'] = config['evaluation']['convergence_profile_sha256']
    if expected != config:
        raise ValueError('Migration may change only name, epoch cap and declared provenance')
    new = copy.deepcopy(old)
    new['config'] = copy.deepcopy(config)
    return new


def train_wolfe(train, valid, classes, recipe, experiment, resume=None):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from came_repro.downstream import score_probe
    from came_repro.experiment import seed_everything, atomic_checkpoint, rng_state, restore_rng
    from came_repro.pause_control import pause_requested, TrainingPaused
    seed_everything(0)
    head = torch.nn.Linear(train.width, classes).cuda()
    opt = torch.optim.LBFGS(head.parameters(), lr=recipe['lr'], max_iter=1, history_size=20,
                           tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    required = len(train)*train.width*4 + 512*2**20
    if required > torch.cuda.mem_get_info()[0]:
        raise RuntimeError('Full train-pair CUDA solver exceeds capacity; no CPU fallback')
    x, y = train.batch(np.arange(len(train)))
    best, selected, stale, first, history, best_state = -1., 0, 0, 1, [], None
    if resume:
        saved = torch.load(str(resume), map_location='cpu', weights_only=False)
        if saved['schema'] != 'came_wolfe_linear_v1' or saved['config'] != experiment.config or saved['data_audit'] != experiment.data_audit:
            raise ValueError('Strong-Wolfe resume identity changed')
        head.load_state_dict(saved['classifier']); opt.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
        best, selected, stale, first, history, best_state = saved['best_accuracy'], saved['selected_epoch'], saved['stale'], saved['next_epoch'], saved['history_rows'], saved['best_state']
        for row in history: experiment.metric(row)
    for epoch in range(first, first if stale >= recipe['patience'] else recipe['max_epochs']+1):
        calls = [0]
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x), y)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite linear CE')
            loss.backward(); calls[0] += 1
            if not all(torch.isfinite(p.grad).all() for p in head.parameters()): raise FloatingPointError('Nonfinite linear gradient')
            return loss
        opt.step(closure)
        closure()
        gradient = float(torch.sqrt(sum(p.grad.square().sum() for p in head.parameters())))
        ce = float(F.cross_entropy(head(x), y).detach())
        accuracy, loss, _ = score_probe(head, valid)
        row = dict(step=epoch, train_loss=ce, gradient_norm=gradient, val_loss=loss, val_accuracy=accuracy, lr=recipe['lr'], closure_evaluations=calls[0])
        history.append(row); experiment.metric(row)
        if accuracy > best:
            best, selected, stale = accuracy, epoch, 0
            best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else: stale += 1
        if epoch == 1 or epoch % 25 == 0:
            experiment.logger.info('Wolfe epoch=%d CE=%.6f gradient=%.6g validation=%.6f selected=%d', epoch, ce, gradient, accuracy, selected)
        atomic_checkpoint(experiment.checkpoint_directory/'probe_last.pt', dict(schema='came_wolfe_linear_v1', classifier=head.state_dict(), optimizer=opt.state_dict(), rng=rng_state(), completed_epochs=epoch, next_epoch=epoch+1, best_accuracy=best, selected_epoch=selected, stale=stale, best_state=best_state, history_rows=history, config=experiment.config, data_audit=experiment.data_audit, cursor_policy='full train-pair outer optimization step; line-search closure count recorded'))
        if pause_requested(): raise TrainingPaused('Strong-Wolfe checkpoint saved')
        if stale >= recipe['patience']: break
    head.load_state_dict(best_state)
    accuracy, loss, predictions = score_probe(head, valid, save_predictions=True)
    if accuracy != best: raise ValueError('Selected validation replay mismatch')
    return dict(validation_accuracy=best, selected_epoch=selected, epochs_executed=len(history), max_epochs=recipe['max_epochs'], patience=recipe['patience'], optimizer='LBFGS_strong_wolfe', lr=recipe['lr'], test_evaluations=0, test_accuracy=None), predictions, best_state, history


def main():
    import json
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.experiment import Experiment, write_json, atomic_checkpoint
    from came_repro.protocol_repair import check_cuda, train_lp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['lp','wn','acceptance'], required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--resume-head', type=Path)
    args = parser.parse_args()
    verify_originals(); check_cuda(); torch.set_num_threads(4)
    profile_path = ROOT/'configs/convergence_repair_20261014.json'
    profile = read_json_file(profile_path)
    if args.stage == 'acceptance':
        from came_repro.experiment import seed_everything
        seed_everything(0)
        x = torch.randn(32, 8, device='cuda'); y = (x[:,0] > 0).long()
        head = torch.nn.Linear(8,2).cuda()
        opt = torch.optim.LBFGS(head.parameters(), lr=.05, max_iter=1, line_search_fn='strong_wolfe', history_size=20)
        before = float(torch.nn.functional.cross_entropy(head(x),y))
        for _ in range(8):
            def closure():
                opt.zero_grad(set_to_none=True); loss=torch.nn.functional.cross_entropy(head(x),y); loss.backward(); return loss
            opt.step(closure)
        after=float(torch.nn.functional.cross_entropy(head(x),y))
        assert np.isfinite(after) and after < before
        # Epoch-budget migration itself may not alter trained weights/optimizer/history/RNG.
        old={'schema':'came_lp_protocol_v1','completed_epochs':2,'next_epoch':3,'stale':0,'config':{'name':'old','evaluation':{'max_epochs':2,'patience':10}},'head':head.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'history_rows':[{'step':1},{'step':2}]}
        config=copy.deepcopy(old['config']);config['name']='new';config['evaluation'].update(max_epochs=3,convergence_profile_sha256='fixture')
        new=migrated_lp_state(old,config,3)
        for key in old['head']: assert torch.equal(old['head'][key],new['head'][key])
        assert torch.equal(old['rng'],new['rng']) and new['history_rows']==old['history_rows'] and old['config']['evaluation']['max_epochs']==2
        class FixtureInput:
            width=8
            def __len__(self): return len(y)
            def batch(self,index): return x[index].clone(),y[index].clone()
        fixture=FixtureInput()
        config={'name':'wolfe_resume_acceptance','seed':0,'storage':read_json_file(ROOT/'configs/storage.json'),'evaluation':{'fixture_only':True,'max_epochs':3}}
        fixture_recipe=dict(profile['WN'],max_epochs=3)
        baseline=Experiment(config,{'fixture_only':True,'not_paper_dataset':True})
        try:
            train_wolfe(fixture,fixture,2,fixture_recipe,baseline)
            baseline.status('completed_acceptance_fixture')
        finally:baseline.close()
        import came_repro.pause_control as pauses
        original_pause=pauses.pause_requested
        interrupted=Experiment(config,{'fixture_only':True,'not_paper_dataset':True})
        # Request a cooperative pause exactly after the second full optimization step.
        counter=[0]
        def pause_at_two():
            counter[0]+=1
            return counter[0]==2
        pauses.pause_requested=pause_at_two
        try:
            try:train_wolfe(fixture,fixture,2,fixture_recipe,interrupted)
            except pauses.TrainingPaused:interrupted.status('paused_acceptance_fixture')
            else:raise ValueError('Acceptance pause did not occur')
        finally:pauses.pause_requested=original_pause;interrupted.close()
        parent=interrupted.checkpoint_directory/'probe_last.pt'
        resumed=Experiment(config,{'fixture_only':True,'not_paper_dataset':True},parent)
        try:
            train_wolfe(fixture,fixture,2,fixture_recipe,resumed,parent)
            resumed.status('completed_acceptance_fixture')
        finally:resumed.close()
        a=torch.load(str(baseline.checkpoint_directory/'probe_last.pt'),map_location='cpu',weights_only=False)
        b=torch.load(str(resumed.checkpoint_directory/'probe_last.pt'),map_location='cpu',weights_only=False)
        assert a['history_rows']==b['history_rows'] and a['next_epoch']==b['next_epoch']==4
        for key in a['classifier']:assert torch.equal(a['classifier'][key],b['classifier'][key])
        acceptance={'passed':True,'actual_torch_and_DGL_CUDA':True,'solver_loss_before':before,'solver_loss_after':after,'budget_migration_keeps_weights_RNG_history':True,'solver_pause_resume_bitexact_weights_and_history':True,'resume_parent_sha256':fingerprint(parent),'fixture_runs':[str(e.directory) for e in [baseline,interrupted,resumed]],'profile_sha256':fingerprint(profile_path),'script_sha256':fingerprint(__file__)}
        write_json(ROOT/'artifacts/audit/convergence_cuda_acceptance_20261014.json',acceptance)
        print(json.dumps(acceptance)); return
    acceptance=read_json_file(ROOT/'artifacts/audit/convergence_cuda_acceptance_20261014.json')
    if not acceptance['passed'] or acceptance['profile_sha256']!=fingerprint(profile_path) or acceptance['script_sha256']!=fingerprint(__file__):
        raise ValueError('Accepted frozen profile and script required')
    reference = read_json_file(args.reference_run/'summary.json')
    cache=Path(reference['embedding_cache']) if 'embedding_cache' in reference else Path(read_json_file(args.reference_run/'config.json')['evaluation']['embedding_cache'])
    model_hash=profile['encoder_sha256'];state=read_json_file(cache/'inference_state.json')
    if not state['complete'] or state['identity']['checkpoint_sha256']!=model_hash or fingerprint(cache/'fused_embeddings.npy')!=state['output_sha256']:
        raise ValueError('Full frozen cache changed')
    if args.stage == 'lp':
        from came_repro.lp_validation_repair import load_verified_cache
        from came_repro.remaining_lp_pilot import use_cached_vectors, verify_row_cache_equivalence
        gpu,audit=load_verified_cache(args.dataset,cache,model_hash)
        config=read_json_file(args.reference_run/'config.json'); recipe=copy.deepcopy(profile['LP'][args.dataset])
        recipe.pop('evidence');recipe.pop('initialization')
        config['name']='convergence_lp_'+args.dataset
        config['evaluation'].update(recipe,convergence_profile_sha256=fingerprint(profile_path))
        previous=None
        if args.dataset=='cloth-copurchase' and not args.resume_head:
            previous=Path(read_json_file(args.reference_run/'checkpoints/location.json')['checkpoint_directory'])/'link_head_last.pt'
        parent=args.resume_head or previous or Path(read_json_file(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
        experiment=Experiment(config,audit,parent)
        try:
            resume=args.resume_head
            if previous:
                old=torch.load(str(previous),map_location='cpu',weights_only=False)
                bridge=migrated_lp_state(old,config,recipe['max_epochs'])
                resume=experiment.checkpoint_directory/'budget_migration_parent.pt'
                atomic_checkpoint(resume,bridge)
                write_json(experiment.directory/'budget_migration_audit.json',{'parent_sha256':fingerprint(previous),'new_bridge_sha256':fingerprint(resume),'old_max_epochs':100,'new_max_epochs':300,'trained_state_unchanged':True,'scientific_budget_changed_not_identical_resume':True})
            write_json(experiment.directory/'cuda_row_cache_equivalence.json',verify_row_cache_equivalence())
            with use_cached_vectors(): summary=train_lp(gpu,torch.load(str(ROOT/'data/raw/mm_graph'/args.dataset/'lp-edge-split.pt'),map_location='cpu',weights_only=False,mmap=True)['train'],torch.load(str(ROOT/'data/raw/mm_graph'/args.dataset/'lp-edge-split.pt'),map_location='cpu',weights_only=False,mmap=True)['valid'],recipe,experiment,resume)
            summary.update(dataset=args.dataset,task='link',seed=0,encoder_sha256=model_hash,embedding_cache=str(cache),graph_policy='upstream_train_bidirected_unique',diagnostic_only=True,paper_reproduction_complete=False)
            write_json(experiment.directory/'summary.json',summary);experiment.status('completed_convergence_validation')
        except BaseException as error:
            from came_repro.pause_control import TrainingPaused
            experiment.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
        finally:experiment.close()
        return
    from came_repro.kg_graph_probe import kg_input
    from came_repro.kg_affine_probe import AffinePairInput,DoublePairInput
    from came_repro.feature_affine import training_pair_statistics,fold_affine_linear
    from came_repro.downstream import score_probe
    from came_repro.prepared_data import prepared_directory
    matrix=np.load(cache/'fused_embeddings.npy',mmap_mode='r')
    manifest=read_json_file(prepared_directory('WN18RR')/'manifest.json')
    for name,digest in manifest['files'].items():
        if fingerprint(prepared_directory('WN18RR')/name)!=digest:raise ValueError('Official KG data changed')
    assert matrix.shape==(40943,1024)
    train_raw=kg_input('WN18RR',matrix,'train'); valid_raw=kg_input('WN18RR',matrix,'valid')
    mean,scale=training_pair_statistics(matrix,train_raw.pairs)
    stats_hash=hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest()
    config=read_json_file(args.reference_run/'config.json');config['name']='wn_wolfe_convergence'
    config['evaluation'].update(profile['WN'],convergence_profile_sha256=fingerprint(profile_path),test_used_for_selection=False)
    audit=copy.deepcopy(state['identity']['data']);audit['feature_affine']={'statistics_split':'official_train_pairs_only','statistics_sha256':stats_hash,'invertible':True}
    encoder=Path(read_json_file(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    experiment=Experiment(config,audit,args.resume_head or encoder)
    try:
        np.savez_compressed(experiment.directory/'train_affine_statistics.npz',mean=mean,scale=scale)
        train=AffinePairInput(matrix,train_raw.labels,pairs=train_raw.pairs).set_affine(mean,scale)
        valid=AffinePairInput(matrix,valid_raw.labels,pairs=valid_raw.pairs).set_affine(mean,scale);valid.cache_validation(400,experiment.logger)
        summary,pred,best,history=train_wolfe(train,valid,manifest['relations'],profile['WN'],experiment,args.resume_head)
        weight,bias=fold_affine_linear(best['weight'].numpy(),best['bias'].numpy(),mean,scale)
        raw=torch.nn.Linear(2048,manifest['relations'],dtype=torch.float64).cuda();raw.load_state_dict({'weight':torch.from_numpy(weight),'bias':torch.from_numpy(bias)})
        accuracy,loss,folded=score_probe(raw,DoublePairInput(matrix,valid_raw.labels,pairs=valid_raw.pairs),save_predictions=True)
        if np.sum(pred!=folded)>2:raise ValueError('Affine folded raw linear equivalence failed')
        summary.update(validation_accuracy=accuracy,validation_loss=loss,dataset='WN18RR',task='relation',seed=0,encoder_sha256=model_hash,embedding_cache=str(cache),graph_policy='train_reverse',feature_transform='train_pair_affine',affine_statistics_sha256=stats_hash,folded_validation_disagreements=int(np.sum(pred!=folded)),edge_representation='head_tail_concatenation',diagnostic_only=True,paper_reproduction_complete=False)
        atomic_checkpoint(experiment.checkpoint_directory/'probe_best_raw_linear.pt',{'classifier':raw.state_dict(),'selected_epoch':summary['selected_epoch'],'config':config,'data_audit':audit})
        np.savez_compressed(experiment.directory/'validation_predictions.npz',prediction=folded,labels=valid_raw.labels,pairs=valid_raw.pairs)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4));steps=[r['step'] for r in history]
        axes[0].plot(steps,[r['train_loss'] for r in history],label='train');axes[0].plot(steps,[r['val_loss'] for r in history],label='validation');axes[0].legend();axes[0].set(xlabel='Outer full-train step',ylabel='Cross entropy')
        axes[1].plot(steps,[r['val_accuracy'] for r in history]);axes[1].axvline(summary['selected_epoch'],ls='--',color='black');axes[1].set(xlabel='Outer full-train step',ylabel='Official validation accuracy')
        fig.text(.02,.01,'WN18RR seed0; strong-Wolfe LBFGS, raw concatenation linear head; test held out.',fontsize=8);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(experiment.directory/'figures/wolfe_probe_curves.png',dpi=180);plt.close(fig)
        write_json(experiment.directory/'summary.json',summary);experiment.status('completed_convergence_validation')
    except BaseException as error:
        from came_repro.pause_control import TrainingPaused
        experiment.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
    finally:experiment.close()


if __name__=='__main__':main()
