"""Bounded validation-only Cloth refinement and WN scheduled linear probe.

Independent infrastructure: historical author files, encoders, datasets and runs stay immutable.
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))


def reference_cache(summary, config):
    """Older validated runs store the same path only in resolved evaluation config."""
    a = summary.get('embedding_cache')
    b = config.get('evaluation', {}).get('embedding_cache')
    if a and b and Path(a).resolve() != Path(b).resolve():
        raise ValueError('Summary/config cache paths disagree')
    if not (a or b): raise ValueError('Validated reference has no cache path')
    return Path(a or b)


def checkpoint_due(task, epoch, maximum, stale, patience, requested):
    """All metrics persist every epoch; checkpoint at safe boundaries without excess disk scans."""
    interval = 25 if task == 'wn' else 5
    return requested or epoch == maximum or stale >= patience or epoch % interval == 0


def checkpoint_payload(head, optimizer, scheduler, random, experiment, task, seed,
                       epoch, history, best, selected, stale, best_state):
    from came_repro.experiment import rng_state
    return {'schema': 'came_priority_refinement_v1', 'task': task, 'seed': seed,
            'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'rng': rng_state(), 'sampling_generator': random.bit_generator.state,
            'config': experiment.config, 'data_audit': experiment.data_audit,
            'completed_epochs': epoch, 'next_epoch': epoch + 1, 'history_rows': history,
            'best_score': best, 'selected_epoch': selected, 'stale': stale,
            'best_state': best_state, 'cursor_policy': 'Complete official train-positive/train-pair epoch boundary'}


def restore_training(path, head, optimizer, scheduler, random, experiment, task, seed):
    import torch
    from came_repro.experiment import restore_rng
    saved = torch.load(str(path), map_location='cpu', weights_only=False)
    if (saved['schema'] != 'came_priority_refinement_v1' or saved['task'] != task or saved['seed'] != seed
            or saved['config'] != experiment.config or saved['data_audit'] != experiment.data_audit
            or saved['next_epoch'] != saved['completed_epochs'] + 1
            or len(saved['history_rows']) != saved['completed_epochs']):
        raise ValueError('Complete matching optimizer/RNG/scheduler/history checkpoint required')
    head.load_state_dict(saved['head']); optimizer.load_state_dict(saved['optimizer'])
    if scheduler: scheduler.load_state_dict(saved['scheduler'])
    elif saved['scheduler'] is not None: raise ValueError('Unexpected saved scheduler')
    restore_rng(saved['rng']); random.bit_generator.state = saved['sampling_generator']
    for row in saved['history_rows']: experiment.metric(row)
    return saved


def train_candidate(task, train, valid, recipe, seed, experiment, resume=None, initial_head=None):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from came_repro.experiment import seed_everything, atomic_checkpoint
    from came_repro.pause_control import pause_requested, TrainingPaused
    from came_repro.downstream import score_probe
    from came_repro.lp_validation_repair import make_head, TrainOnlyNegatives
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro.task_metrics import reverse_overlap_mask
    from came_repro import link_evaluation as link
    from run_lp_hard_negative_candidate import sample_hard_negative
    seed_everything(seed); random = np.random.default_rng(seed)
    if task == 'cloth':
        gpu, split = train
        source, target = (np.asarray(split[key], dtype=np.int64) for key in ('source_node', 'target_node'))
        sampler = TrainOnlyNegatives(np.concatenate((source, target)), np.concatenate((target, source)), len(gpu))
        overlap = reverse_overlap_mask(source, target, np.asarray(valid['source_node']), np.asarray(valid['target_node']), len(gpu))
        head = make_head(recipe['head_variant']).cuda()
        if initial_head is None: raise ValueError('Cloth refinement requires a frozen validation-selected initialization')
        head.load_state_dict(initial_head)
        x = y = None
    else:
        head = torch.nn.Linear(train.width, recipe['classes']).cuda()
        if len(train) * train.width * 4 + 512 * 2**20 > torch.cuda.mem_get_info()[0]:
            raise RuntimeError('Full WN train-pair CUDA cache exceeds capacity; no CPU fallback')
        x, y = train.batch(np.arange(len(train)))
    optimizer = torch.optim.Adam(head.parameters(), lr=recipe['lr'])
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.1,
        patience=recipe['lr_patience'], threshold=0., min_lr=recipe['min_lr']) if task == 'wn' else None)
    first, history, best, selected, stale, best_state = 1, [], -1., 0, 0, None
    if resume:
        saved = restore_training(resume, head, optimizer, scheduler, random, experiment, task, seed)
        first, history, best, selected, stale, best_state = (saved[k] for k in
            ('next_epoch', 'history_rows', 'best_score', 'selected_epoch', 'stale', 'best_state'))
    for epoch in range(first, first if stale >= recipe['patience'] else recipe['max_epochs'] + 1):
        head.train(); lr = optimizer.param_groups[0]['lr']; total = 0.; hard_total = 0.
        if task == 'cloth':
            with use_cached_vectors():
                order = random.permutation(len(source))
                for start in range(0, len(order), recipe['batch_size']):
                    ids = order[start:start + recipe['batch_size']]
                    negative, pool, scores = sample_hard_negative(head, gpu, source[ids], sampler, random)
                    hard_total += float(scores.max(1).values.mean()) * len(ids)
                    a, b, c = (link.vectors(gpu, value) for value in (source[ids], target[ids], negative))
                    optimizer.zero_grad(set_to_none=True)
                    positive, neg = head(a, b), head(a, c)
                    loss = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive)) + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite Cloth BCE')
                    loss.backward()
                    if not all(torch.isfinite(p.grad).all() for p in head.parameters()): raise FloatingPointError('Nonfinite Cloth gradient')
                    optimizer.step(); total += float(loss) * len(ids)
                value = link.evaluate_split(head, gpu, valid, 64, overlap)['official']['mrr']
            row = {'step': epoch, 'train_loss': total / len(source), 'validation_mrr': value,
                   'lr_used': lr, 'lr_next': optimizer.param_groups[0]['lr'],
                   'negative_pool_size': 4, 'mean_selected_negative_score': hard_total / len(source)}
        else:
            optimizer.zero_grad(set_to_none=True); loss = F.cross_entropy(head(x), y)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite WN CE')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in head.parameters()): raise FloatingPointError('Nonfinite WN gradient')
            optimizer.step()
            value, val_loss, _ = score_probe(head, valid)
            scheduler.step(val_loss)
            row = {'step': epoch, 'train_loss': float(loss), 'val_loss': val_loss, 'val_accuracy': value,
                   'lr_used': lr, 'lr_next': optimizer.param_groups[0]['lr']}
        history.append(row); experiment.metric(row)
        if value > best:
            best, selected, stale = value, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else: stale += 1
        experiment.logger.info('Priority %s seed=%d epoch=%d loss=%.7f validation=%.9f selected=%d lr=%.8g', task, seed, epoch, row['train_loss'], value, selected, row['lr_next'])
        requested = pause_requested()
        if checkpoint_due(task, epoch, recipe['max_epochs'], stale, recipe['patience'], requested):
            atomic_checkpoint(experiment.checkpoint_directory / 'priority_head_last.pt',
                checkpoint_payload(head, optimizer, scheduler, random, experiment, task, seed, epoch, history, best, selected, stale, best_state))
        if requested: raise TrainingPaused('Priority complete epoch checkpoint saved')
        if stale >= recipe['patience']: break
    head.load_state_dict(best_state)
    if task == 'cloth':
        with use_cached_vectors(): replay = link.evaluate_split(head, gpu, valid, 64, overlap, experiment.directory / 'validation_link_ranks.npz')['official']['mrr']
        if abs(replay - best) > 1e-12: raise ValueError('Cloth saved validation replay mismatch')
        atomic_checkpoint(experiment.checkpoint_directory / 'link_head_best.pt', {'head': best_state, 'selected_epoch': selected, 'validation_mrr': best})
        summary = dict(recipe, seed=seed, validation_mrr=best, selected_epoch=selected, epochs_executed=len(history),
                       test_evaluations=0, test=None, negative_pool_size=4, negative_policy='max_score_train_and_reverse_known_nonedge')
    else:
        replay, _, _ = score_probe(head, valid)
        if replay != best: raise ValueError('WN saved validation replay mismatch')
        summary = dict(recipe, seed=seed, validation_accuracy=best, selected_epoch=selected, epochs_executed=len(history), test_accuracy=None, test_evaluations=0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['step'] for r in history], [r['train_loss'] for r in history])
    key = 'validation_mrr' if task == 'cloth' else 'val_accuracy'
    axes[1].plot([r['step'] for r in history], [r[key] for r in history]); axes[1].axvline(selected, color='black', ls='--')
    for axis in axes: axis.set_xlabel('Complete refinement/probe epoch')
    axes[0].set_ylabel('Train BCE/CE'); axes[1].set_ylabel('Official validation MRR/accuracy')
    fig.text(.02, .01, f'{task} seed{seed}; frozen encoder, full official splits, validation selected; test held out.', fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(experiment.directory / 'figures/hard_lp_curves.png' if task == 'cloth' else experiment.directory / 'figures/priority_probe_curves.png', dpi=180); plt.close(fig)
    return summary, best_state, history


def acceptance(profile_path):
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro.experiment import Experiment, seed_everything
    from came_repro.downstream import ProbeInput
    from came_repro.lp_validation_repair import make_head
    import came_repro.pause_control as pauses
    profile = read_json_file(profile_path); seed_everything(191)
    gpu = torch.randn(12, 1024, device='cuda')
    initial = {k: v.detach().cpu().clone() for k, v in make_head(profile['cloth']['recipe']['head_variant']).state_dict().items()}
    source = np.arange(6, dtype=np.int64)
    valid_lp = {'source_node': np.array([6, 7]), 'target_node': np.array([7, 8]), 'target_node_neg': np.tile(np.arange(12), (2, 13))[:, :150]}
    features = np.random.default_rng(1).normal(size=(20, 8)).astype(np.float32)
    probe = ProbeInput(features, np.arange(20) % 3, indices=np.arange(20))
    reports = {}
    for task in ('cloth', 'wn'):
        recipe = copy.deepcopy(profile[task]['recipe']); recipe.update(max_epochs=3, patience=50)
        if task == 'cloth': recipe['batch_size'] = 3
        else: recipe.update(classes=3, lr_patience=0)
        train = (gpu, {'source_node': source, 'target_node': source + 1}) if task == 'cloth' else probe
        valid = valid_lp if task == 'cloth' else probe
        config = {'name': 'priority_resume_' + task, 'seed': 1, 'storage': read_json_file(ROOT / 'configs/storage.json'), 'evaluation': {'fixture_only': True, 'task': task, 'recipe': recipe}}
        runs = []
        for stage in ('baseline', 'pause', 'resume'):
            parent = runs[-1].checkpoint_directory / 'priority_head_last.pt' if stage == 'resume' else None
            experiment = Experiment(config, {'fixture_only': True}, parent); runs.append(experiment)
            original = pauses.pause_requested; calls = [0]
            def after_two():
                calls[0] += 1
                return calls[0] == 2
            if stage == 'pause': pauses.pause_requested = after_two
            try:
                train_candidate(task, train, valid, recipe, 1, experiment, parent, initial if task == 'cloth' else None)
                if stage == 'pause': raise ValueError('Expected acceptance pause')
                experiment.status('completed_priority_fixture')
            except pauses.TrainingPaused:
                if stage != 'pause': raise
                experiment.status('paused_priority_fixture')
            finally:
                pauses.pause_requested = original; experiment.close()
        a = torch.load(str(runs[0].checkpoint_directory / 'priority_head_last.pt'), map_location='cpu', weights_only=False)
        b = torch.load(str(runs[2].checkpoint_directory / 'priority_head_last.pt'), map_location='cpu', weights_only=False)
        def equal(x, y):
            if isinstance(x, torch.Tensor): return torch.equal(x, y)
            if isinstance(x, dict): return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
            if isinstance(x, (list, tuple)): return len(x) == len(y) and all(equal(v, w) for v, w in zip(x, y))
            return x == y
        for key in ('head', 'optimizer', 'scheduler', 'history_rows', 'next_epoch', 'best_state'):
            assert equal(a[key], b[key]), task + ' resume mismatch: ' + key
        reports[task] = {'seed1_optimizer_scheduler_history_weights_bitexact': True, 'fixture_runs': [str(r.directory) for r in runs]}
    report = {'passed': True, 'actual_Torch_and_DGL_CUDA': True, 'fixture_not_benchmark': True,
              'tasks': reports, 'profile_sha256': fingerprint(profile_path), 'script_sha256': fingerprint(__file__)}
    atomic_write_json(ROOT / 'artifacts/audit/priority_cloth_wn_CUDA_acceptance_20261016.json', report)
    print(__import__('json').dumps(report))


def main():
    import hashlib
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.protocol_repair import check_cuda
    from came_repro.experiment import Experiment, atomic_checkpoint
    from came_repro.pause_control import TrainingPaused
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['acceptance', 'cloth', 'wn'], required=True)
    parser.add_argument('--resume-head', type=Path)
    args = parser.parse_args(); verify_originals(); check_cuda(); torch.set_num_threads(4)
    profile_path = ROOT / 'configs/priority_cloth_wn_20261016.json'
    profile = read_json_file(profile_path)
    if args.task == 'acceptance': acceptance(profile_path); return
    accepted = read_json_file(ROOT / 'artifacts/audit/priority_cloth_wn_CUDA_acceptance_20261016.json')
    if not accepted['passed'] or accepted['script_sha256'] != fingerprint(__file__) or accepted['profile_sha256'] != fingerprint(profile_path): raise ValueError('Actual frozen CUDA/resume acceptance required')
    item = profile[args.task]; reference = ROOT / item['reference_run']
    if fingerprint(reference / 'summary.json') != item['reference_summary_sha256']: raise ValueError('Frozen validation reference changed')
    old = read_json_file(reference / 'summary.json')
    reference_config = read_json_file(reference / 'config.json')
    cache = reference_cache(old, reference_config)
    state = read_json_file(cache / 'inference_state.json')
    if not state['complete'] or state['identity']['checkpoint_sha256'] != profile['encoder_sha256'] or fingerprint(cache / 'fused_embeddings.npy') != state['output_sha256']: raise ValueError('Complete frozen cache identity required')
    config = copy.deepcopy(reference_config); config.update(name='priority_' + args.task, seed=0)
    config['evaluation'].update(item['recipe'], profile_sha256=fingerprint(profile_path), backend_sha256=fingerprint(__file__), test_used_for_selection=False)
    config['evaluation'].update(optimizer='Adam', weight_decay=0.,
        scheduler='ReduceLROnPlateau' if args.task == 'wn' else None,
        scheduler_mode='min' if args.task == 'wn' else None,
        scheduler_factor=.1 if args.task == 'wn' else None,
        scheduler_threshold=0. if args.task == 'wn' else None,
        checkpoint_interval_epochs=25 if args.task == 'wn' else 5,
        checkpoint_also_on_pause_stop_and_final=True,
        train_and_validation_complete=True, official_test_held_out=True)
    audit = copy.deepcopy(state['identity']['data'])
    initial = None
    if args.task == 'cloth':
        from came_repro.lp_validation_repair import load_verified_cache
        gpu, audit = load_verified_cache('cloth-copurchase', cache, profile['encoder_sha256'])
        if len(gpu) != 125839 or audit.get('diagnostic_topology', {}).get('policy') != 'upstream_train_bidirected_unique': raise ValueError('Exact full reference Cloth train-only graph required')
        parent = Path(read_json_file(reference / 'checkpoints/location.json')['checkpoint_directory']) / 'link_head_best.pt'
        if fingerprint(parent) != item['initial_best_head_sha256']: raise ValueError('Cloth warm-start head changed')
        initial = torch.load(str(parent), map_location='cpu', weights_only=False)['head']
        raw = ROOT / 'data/raw/mm_graph/cloth-copurchase/lp-edge-split.pt'
        if fingerprint(raw) != audit['raw_sha256']['lp-edge-split.pt']: raise ValueError('Official Cloth split changed')
        split = torch.load(str(raw), map_location='cpu', weights_only=False, mmap=True)
        if len(split['train']['source_node']) != 951271 or np.asarray(split['valid']['target_node_neg']).shape != (121323, 150): raise ValueError('Complete official Cloth train/validation required')
        train, valid = (gpu, split['train']), split['valid']
        config['evaluation'].update(initialization='new_optimizer_from_previous_validation_best_not_identical_resume', initialization_sha256=fingerprint(parent))
    else:
        from came_repro.kg_graph_probe import kg_input
        from came_repro.kg_affine_probe import AffinePairInput, DoublePairInput
        from came_repro.feature_affine import training_pair_statistics, fold_affine_linear
        from came_repro.prepared_data import prepared_directory
        manifest = read_json_file(prepared_directory('WN18RR') / 'manifest.json')
        for name, digest in manifest['files'].items():
            if fingerprint(prepared_directory('WN18RR') / name) != digest: raise ValueError('Official WN data changed')
        matrix = np.load(cache / 'fused_embeddings.npy', mmap_mode='r')
        if matrix.shape != (40943, 1024): raise ValueError('Full WN1024 cache required')
        raw_train, raw_valid = kg_input('WN18RR', matrix, 'train'), kg_input('WN18RR', matrix, 'valid')
        mean, scale = training_pair_statistics(matrix, raw_train.pairs)
        statistics_sha = hashlib.sha256(mean.tobytes() + scale.tobytes()).hexdigest()
        if statistics_sha != old['affine_statistics_sha256']: raise ValueError('Train-only WN affine statistics changed')
        audit['feature_affine'] = {'statistics_split': 'official_train_pairs_only', 'statistics_sha256': statistics_sha, 'invertible': True}
        train = AffinePairInput(matrix, raw_train.labels, pairs=raw_train.pairs).set_affine(mean, scale)
        valid = AffinePairInput(matrix, raw_valid.labels, pairs=raw_valid.pairs).set_affine(mean, scale)
        parent = Path(read_json_file(ROOT / 'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
        if fingerprint(parent) != profile['encoder_sha256']: raise ValueError('Protected encoder changed')
        config['evaluation']['initialization'] = 'fixed_seed0_fresh_linear_head'
    experiment = Experiment(config, audit, args.resume_head or parent)
    try:
        if args.task == 'wn': valid.cache_validation(400, experiment.logger)
        summary, best, history = train_candidate(args.task, train, valid, item['recipe'], 0, experiment, args.resume_head, initial)
        summary.update(dataset='cloth-copurchase' if args.task == 'cloth' else 'WN18RR', encoder_sha256=profile['encoder_sha256'], embedding_cache=str(cache),
                       graph_policy=old['graph_policy'], diagnostic_only=True, paper_reproduction_complete=False, optimizer_adaptation_not_author_confirmed=True)
        if args.task == 'wn':
            from came_repro.downstream import score_probe
            weight, bias = fold_affine_linear(best['weight'].numpy(), best['bias'].numpy(), mean, scale)
            raw = torch.nn.Linear(2048, 11, dtype=torch.float64).cuda(); raw.load_state_dict({'weight': torch.from_numpy(weight), 'bias': torch.from_numpy(bias)})
            value, loss, pred = score_probe(raw, DoublePairInput(matrix, raw_valid.labels, pairs=raw_valid.pairs), save_predictions=True)
            if value != summary['validation_accuracy']: raise ValueError('Folded linear validation changed')
            summary.update(validation_loss=loss, feature_transform='train_pair_affine', affine_statistics_sha256=statistics_sha, edge_representation='head_tail_concatenation')
            atomic_checkpoint(experiment.checkpoint_directory / 'probe_best_raw_linear.pt', {'classifier': raw.state_dict(), 'selected_epoch': summary['selected_epoch'], 'config': config, 'data_audit': audit})
            np.savez_compressed(experiment.directory / 'validation_predictions.npz', prediction=pred, labels=raw_valid.labels, pairs=raw_valid.pairs)
        atomic_write_json(experiment.directory / 'summary.json', summary); experiment.status('completed_priority_validation_only')
    except BaseException as error:
        experiment.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error)); raise
    finally: experiment.close()


if __name__ == '__main__': main()
