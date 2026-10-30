"""One explicitly bounded representation-budget experiment, isolated from old caches."""
import argparse
import copy
import gc
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
PLAN = ROOT / 'configs/representation_coverage_validation_20261016.json'
CONFIG = ROOT / 'configs/representation_coverage_20261016.json'
DIRECTORY = ROOT / 'artifacts/representation_coverage_20261016'
NAMESPACE = DIRECTORY / 'ppr_namespace'


def validate_single_change(original, candidate):
    a, b = copy.deepcopy(original), copy.deepcopy(candidate)
    for key in ('name', 'deviations'):
        a.pop(key); b.pop(key)
    old = a['sampling'].pop('seed_limits')
    new = b['sampling'].pop('seed_limits')
    if a != b or any(new[k] != 2 * v for k, v in old.items()) or set(old) != set(new):
        raise ValueError('Only fixed unlabeled pool sizes may change numerically')
    if b['training']['epochs'] != 8 or b['training']['max_steps'] is not None:
        raise ValueError('Require exactly eight declared subset rounds, no truncated candidate')


def logical_digest(connection):
    digest = hashlib.sha256(); count = 0
    for seed, nodes, residual in connection.execute('SELECT seed,nodes,residual FROM ppr ORDER BY seed'):
        digest.update(str(seed).encode() + b'\0' + nodes + b'\0' + float(residual).hex().encode() + b'\n')
        count += 1
    return count, digest.hexdigest()


def clone_readonly_cache(source, target):
    """SQLite backup from a read-only historical DB; validate every logical row."""
    from came_repro.author_loader import fingerprint
    source = Path(source).resolve(); target = Path(target).resolve()
    if source == target or target.is_relative_to(source.parent):
        raise ValueError('Private PPR cache must be outside the original cache directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    before = fingerprint(source)
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as original:
        expected = logical_digest(original)
        if not target.exists():
            with sqlite3.connect(str(target)) as private:
                original.backup(private)
        with sqlite3.connect(target.as_uri() + '?mode=ro', uri=True) as private:
            actual = logical_digest(private)
        if actual != expected:
            raise ValueError('Initial private clone differs from historical cache')
    if fingerprint(source) != before:
        raise ValueError('Historical PPR database changed during read-only cloning')
    return {'source': str(source), 'private': str(target), 'source_sha256': before,
            'logical_rows': expected[0], 'logical_sha256': expected[1],
            'historical_cache_modified': False}


def prepare():
    import torch
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.storage_budget import check_budget
    from came_repro.subset_offline_train import FixedSubsetSchedule
    verify_originals()
    original = read_json_file(ROOT / 'configs/subset_offline_24h.json')
    config = copy.deepcopy(original)
    config['name'] = 'coverage_pool2x_eight_subset_rounds'
    config['sampling']['seed_limits'] = {k: 2 * v for k, v in original['sampling']['seed_limits'].items()}
    config['deviations'] = [
        'Single finite representation-budget diagnostic. Pool sizes are reconstruction choices, not disclosed author settings.',
        'Fresh seed42 initialization; exactly eight subset rounds, 2560 batches per round, 20480 updates. No old-encoder warm start and no full-graph epoch claim.',
        'Only fixed uniform unlabeled seed pool sizes doubled versus the original declared profile; architecture, PPR, batch32, losses, LR and all numerical training settings unchanged.',
        'Historical PPR databases copied read-only into a private namespace; full immutable prepared topology/features retained.',
        'Cloth and WN seed0 full validation only; at least +0.5pp over the strongest available prior validation result before frozen-weight full test once.',
        'This new backbone is for unresolved tasks only. Six completed formal task results keep their original backbone. A common-backbone full-paper reproduction is not claimed.',
        'Windows/cu118, clipping1.0, unknown CLIP/PPR/mixing and unreleased author loader/head details remain explicit engineering uncertainties.']
    validate_single_change(original, config)
    if CONFIG.exists() and read_json_file(CONFIG) != config:
        raise ValueError('Already declared candidate configuration differs')
    atomic_write_json(CONFIG, config)
    budget = check_budget(additional_project_bytes=512 * 2**20, additional_cache_bytes=4 * 2**30)
    counts = {'ogbn-products': 2449029, 'books-lp': 636502, 'cloth-copurchase': 125839}
    schedule = FixedSubsetSchedule(counts, 32, torch.Generator().manual_seed(42), config['sampling']['seed_limits'])
    review = read_json_file(ROOT / 'artifacts/audit/pretraining_context_coverage_review_20261014.json')
    old_run = ROOT / 'runs/20261011T105355Z_suite_pretra_0c927d_s42_9521d7'
    import numpy as np
    pools, clones = {}, {}
    for row in review['datasets']:
        name = row['dataset']; old_pool = np.load(old_run / ('sampling/' + name + '_seed_pool.npy'))
        new_pool = schedule.pools[name].numpy()
        if not np.array_equal(old_pool, new_pool[:len(old_pool)]) or len(np.unique(new_pool)) != 2 * len(old_pool):
            raise ValueError('The original unlabeled pool is not an exact prefix of the enlarged pool')
        manifest_path = Path(row['cache_manifest']); destination = NAMESPACE / 'data/processed/ppr_packed' / manifest_path.parent.name
        destination.mkdir(parents=True, exist_ok=True)
        identity = read_json_file(manifest_path)
        if identity['nodes'] != counts[name]: raise ValueError('PPR cache node identity mismatch')
        target_manifest = destination / 'manifest.json'
        if target_manifest.exists() and read_json_file(target_manifest) != identity:
            raise ValueError('Private PPR cache identity changed')
        atomic_write_json(target_manifest, identity)
        clones[name] = clone_readonly_cache(manifest_path.parent / 'neighbors.sqlite3', destination / 'neighbors.sqlite3')
        path = DIRECTORY / 'seed_pools' / (name + '.npy'); path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, new_pool)
        pools[name] = {'count': len(new_pool), 'old_prefix_count': len(old_pool), 'sha256': fingerprint(path), 'original_prefix_identical': True}
    references = {}
    for name, relative in {'cloth': 'runs/20261015T233708Z_priority_clo_73daaf_s0_a2c6c7',
                           'wn': 'runs/20261014T045920Z_kg_affine_WN_fce445_s0_cb5aeb'}.items():
        run = ROOT / relative
        references[name] = {'run': str(run), 'summary_sha256': fingerprint(run / 'summary.json')}
    plan = {'stage': 'one_fresh_joint_representation_coverage_candidate', 'configuration_sha256': fingerprint(CONFIG),
            'fresh_initialization': True, 'pretraining_seed': 42, 'downstream_seed': 0,
            'expected_updates': 20480, 'subset_rounds': 8, 'full_graph_epochs_claimed': 0,
            'max_representation_candidates': 1, 'maximum_validation_candidates_per_task': 1,
            'minimum_validation_gain_pp': .5, 'goal_abs_gap_pp': 2., 'score_guaranteed': False,
            'reference_validation': {'cloth': .3025347366384903, 'wn': .7982860909690178},
            'reference_note': 'Cloth uses the stronger completed refinement validation; WN uses the stronger original affine validation. No test chooses models.',
            'references': references, 'WN': read_json_file(ROOT / 'configs/protocol_repair_20261014.json')['WN'],
            'cloth_recipe': dict(read_json_file(ROOT / 'configs/cloth_predictor_structure_20261015.json')['matched_recipe'], head_variant='two_hidden_layernorm_head_upstream_schedule'),
            'test_used_for_selection': False, 'formal_repeats_started': 0,
            'private_ppr_root': str(NAMESPACE), 'resource_reservation': budget,
            'source_evidence': {'CAME': 'https://proceedings.mlr.press/v306/he26at.html',
                'paper_local_pages': [14, 16], 'author_subset_counts_published': False,
                'upstream_MMGraph_revision': '0769fa670c69a6dc0756a2d7fc697b980e494775',
                'OFA_revision': '9045ccdfa2d02fd55841700e7d72ef563bbc2de9'},
            'unknowns': ['CLIP exact checkpoint semantics', 'CAME subset counts/mixing/PPR hyperparameters', 'CAME exact graph and LP loader/head'],
            'protected': ['all six formal task results', 'all original author files', 'old encoder', 'historical raw data/runs/caches', 'Products committed prefix'],
            'clones': clones, 'pools': pools}
    if PLAN.exists(): raise ValueError('Plan already declared; audit it rather than overwrite')
    atomic_write_json(PLAN, plan)
    print(json.dumps({'prepared': True, 'config': str(CONFIG), 'plan': str(PLAN), 'budget': budget}))


def namespace_ppr():
    import came_repro.joint_sampling as sampling
    sampling.ROOT = NAMESPACE


def calibration():
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.experiment import seed_everything
    from came_repro.protocol_repair import check_cuda
    from came_repro.joint_sampling import PackedPPRSampler
    from came_repro.prepared_data import load_prepared
    from came_repro.train import make_model
    verify_originals(); check_cuda(); torch.set_num_threads(4); seed_everything(42); namespace_ppr()
    config = read_json_file(CONFIG); model = make_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    rows = []
    for name in config['datasets']:
        data = load_prepared(name, require_verified=True); sampler = PackedPPRSampler(data, config['sampling'])
        try:
            pool = np.load(DIRECTORY / 'seed_pools' / (name + '.npy'))
            start = time.perf_counter(); graph, nodes, residual = sampler.sample(pool[len(pool)//2:len(pool)//2+32].tolist())
            sampling_seconds = time.perf_counter() - start
            a, b = data.modality_batch(nodes); graph, a, b = graph.to('cuda'), a.cuda(), b.cuda()
            optimizer.zero_grad(set_to_none=True); model.train(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            start = time.perf_counter(); loss, outputs = model(graph, a, b, return_embeddings=True)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite real-corpus acceptance loss')
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError('Nonfinite CUDA gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            row = {'dataset': name, 'seed_batch_size': 32, 'pool_segment': 'first32_new_unlabeled_pool_tail',
                   'nodes': len(nodes), 'edges': graph.num_edges(), 'loss': float(loss),
                   'sampling_seconds': sampling_seconds, 'training_seconds': time.perf_counter()-start,
                   'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                   'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20, 'ppr_residual': residual}
            rows.append(row); print(json.dumps(row), flush=True)
            del outputs, loss, graph, a, b
        finally:
            sampler.close(); del sampler, data; gc.collect(); torch.cuda.empty_cache()
    atomic_write_json(DIRECTORY / 'CUDA_capacity_acceptance.json', {'passed': True, 'actual_Torch_and_DGL_CUDA': True,
        'discarded_calibration_weights_not_candidate_pretraining': True, 'all_widths_retained': 1024,
        'profile_sha256': fingerprint(PLAN), 'configuration_sha256': fingerprint(CONFIG),
        'backend_sha256': fingerprint(__file__), 'examples': rows,
        'small_sample_capacity_not_all_batches_guarantee': True})


def resume_acceptance():
    import numpy as np
    import torch
    import dgl
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro.experiment import Experiment, seed_everything, rng_state, restore_rng
    from came_repro.protocol_repair import check_cuda
    from came_repro.subset_offline_train import FixedSubsetSchedule, save_subset_checkpoint
    from came_repro.train import make_model
    check_cuda(); torch.set_num_threads(4); seed_everything(17)
    config = read_json_file(CONFIG); config.update(name='coverage_joint_resume_fixture', seed=17,
        storage=read_json_file(ROOT / 'configs/storage.json'))
    audit = {'fixture_only_not_benchmark': True, 'graph_nodes': 8, 'actual_author_model': True}
    graph = dgl.graph((torch.arange(8), (torch.arange(8)+1) % 8), num_nodes=8).to('cuda')
    a, b = torch.randn(8, 512, device='cuda'), torch.randn(8, 512, device='cuda')
    counts = {name: 8 for name in config['datasets']}; limits = {name: 4 for name in counts}
    def schedule(): return FixedSubsetSchedule(counts, 2, torch.Generator().manual_seed(17), limits)
    def update(model, optimizer):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = model(graph, a, b)
        if not torch.isfinite(loss): raise FloatingPointError('Fixture loss is nonfinite')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return float(loss)
    def cpu_copy(value):
        if torch.is_tensor(value): return value.detach().cpu().clone()
        if isinstance(value, dict): return {k: cpu_copy(v) for k, v in value.items()}
        if isinstance(value, list): return [cpu_copy(v) for v in value]
        return copy.deepcopy(value)
    model = make_model(config); optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    sampling = schedule(); baseline = Experiment(config, audit)
    try:
        sampling.next_batch(); first_loss = update(model, optimizer)
        baseline.metric({'step': 1, 'loss': first_loss})
        parent = baseline.checkpoint_directory / 'fixture_joint_step1.pt'
        save_subset_checkpoint(parent, {'schema': 'came_joint_v1', 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'schedule': sampling.state_dict(),
            'config': config, 'data_audit': audit, 'step': 1, 'counts': {'fixture': 1}})
        expected_next = sampling.next_batch(); second_loss = update(model, optimizer)
        baseline.metric({'step': 2, 'loss': second_loss}); expected_rng = rng_state()
        expected_model, expected_optimizer = cpu_copy(model.state_dict()), cpu_copy(optimizer.state_dict())
        expected_schedule = sampling.state_dict()
        baseline.status('completed_joint_resume_fixture_reference')
    finally: baseline.close()
    del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    seed_everything(999)
    saved = torch.load(str(parent), map_location='cpu', weights_only=False)
    model = make_model(config); model.load_state_dict(saved['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001); optimizer.load_state_dict(saved['optimizer'])
    sampling = schedule(); sampling.load_state_dict(saved['schedule']); restore_rng(saved['rng'])
    resumed = Experiment(config, audit, parent)
    try:
        # Experiment construction must not perturb the saved training RNG.
        restore_rng(saved['rng'])
        resumed.metric({'step': 1, 'loss': first_loss})
        actual_next = sampling.next_batch(); assert expected_next[0] == actual_next[0] and torch.equal(expected_next[1], actual_next[1])
        replay_loss = update(model, optimizer); resumed.metric({'step': 2, 'loss': replay_loss})
        maximum = 0.
        def compare(actual, expected):
            nonlocal maximum
            if torch.is_tensor(expected):
                actual = actual.detach().cpu()
                if expected.is_floating_point():
                    maximum = max(maximum, float((actual - expected).abs().max()) if expected.numel() else 0.)
                    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-6)
                else: assert torch.equal(actual, expected)
            elif isinstance(expected, dict):
                assert actual.keys() == expected.keys()
                for k in expected: compare(actual[k], expected[k])
            elif isinstance(expected, (list, tuple)):
                assert len(actual) == len(expected)
                for x, y in zip(actual, expected): compare(x, y)
            elif isinstance(expected, np.ndarray): np.testing.assert_array_equal(actual, expected)
            else: assert actual == expected
        compare(model.state_dict(), expected_model); compare(optimizer.state_dict(), expected_optimizer)
        compare(sampling.state_dict(), expected_schedule); compare(rng_state(), expected_rng)
        assert abs(replay_loss - second_loss) <= 1e-5
        resumed.status('completed_joint_resume_acceptance')
        atomic_write_json(DIRECTORY / 'joint_resume_acceptance.json', {'passed': True,
            'actual_Torch_and_DGL_CUDA': True, 'fixture_only_not_benchmark': True,
            'next_dataset_seed_batch_orders_cursors_RNG_verified': True, 'model_optimizer_verified': True,
            'max_model_optimizer_abs_difference': maximum, 'rtol': 1e-4, 'atol': 1e-6,
            'original_second_loss': second_loss, 'resumed_second_loss': replay_loss,
            'parent_checkpoint_sha256': fingerprint(parent), 'reference_run': str(baseline.directory),
            'resumed_run': str(resumed.directory), 'backend_sha256': fingerprint(__file__),
            'profile_sha256': fingerprint(PLAN)})
    finally: resumed.close()


def pretrain(resume=None):
    from came_repro.atomic_io import read_json_file
    from came_repro.author_loader import fingerprint
    from came_repro import subset_offline_train
    accepted = read_json_file(DIRECTORY / 'CUDA_capacity_acceptance.json')
    if (not accepted['passed'] or accepted['backend_sha256'] != fingerprint(__file__)
            or accepted['profile_sha256'] != fingerprint(PLAN) or accepted['configuration_sha256'] != fingerprint(CONFIG)):
        raise ValueError('Matching actual CUDA capacity acceptance required')
    restored = read_json_file(DIRECTORY / 'joint_resume_acceptance.json')
    if not restored['passed'] or restored['backend_sha256'] != fingerprint(__file__) or restored['profile_sha256'] != fingerprint(PLAN):
        raise ValueError('Matching actual author-model optimizer/RNG/sampling restoration acceptance required')
    namespace_ppr()
    sys.argv = ['subset_offline_train', '--config', str(CONFIG)] + (['--resume', str(resume)] if resume else [])
    subset_offline_train.main()


def encode(checkpoint, dataset):
    from types import SimpleNamespace
    from came_repro import protocol_repair
    from came_repro.atomic_io import read_json_file
    protocol_repair.PROFILE = PLAN
    protocol_repair.check_cuda()
    protocol_repair.encode(SimpleNamespace(checkpoint=checkpoint, dataset=dataset), read_json_file(PLAN))


def probe_cloth(checkpoint, resume):
    import torch
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.experiment import Experiment
    from came_repro.protocol_repair import candidate_cache, check_cuda
    from came_repro.lp_validation_repair import load_verified_cache
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro.pause_control import TrainingPaused
    from run_lp_hard_negative_candidate import train_hard
    check_cuda(); torch.set_num_threads(4)
    storage = read_json_file(ROOT / 'configs/storage.json'); plan = read_json_file(PLAN)
    digest = fingerprint(checkpoint); cache = candidate_cache('cloth-copurchase', digest, storage)
    gpu, audit = load_verified_cache('cloth-copurchase', cache, digest)
    if audit.get('diagnostic_topology', {}).get('policy') != 'upstream_train_bidirected_unique':
        raise ValueError('Full official train-only bidirected graph required')
    config = copy.deepcopy(torch.load(str(checkpoint), map_location='cpu', weights_only=False)['config'])
    config.update(name='coverage_cloth_probe', seed=0, storage=storage)
    recipe = plan['cloth_recipe']
    config['evaluation'] = dict(recipe, dataset='cloth-copurchase', task='link', seed=0, encoder_sha256=digest,
        embedding_cache=str(cache), negative_pool_size=4, plan_sha256=fingerprint(PLAN), test_used_for_selection=False)
    experiment = Experiment(config, audit, resume or checkpoint)
    try:
        raw = ROOT / 'data/raw/mm_graph/cloth-copurchase/lp-edge-split.pt'
        if fingerprint(raw) != audit['raw_sha256']['lp-edge-split.pt']: raise ValueError('Official raw split changed')
        split = torch.load(str(raw), map_location='cpu', weights_only=False, mmap=True)
        with use_cached_vectors(): summary = train_hard(gpu, split['train'], split['valid'], recipe, 0, experiment, resume)
        summary.update(dataset='cloth-copurchase', task='link', representation='came', encoder_sha256=digest,
                       embedding_cache=str(cache), graph_policy='upstream_train_bidirected_unique',
                       diagnostic_only=True, paper_reproduction_complete=False)
        atomic_write_json(experiment.directory / 'summary.json', summary)
        experiment.status('completed_coverage_cloth_validation_only')
    except BaseException as error:
        experiment.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error)); raise
    finally: experiment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'calibration', 'resume_acceptance', 'pretrain', 'encode_cloth', 'encode_wn', 'probe_cloth', 'probe_wn'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume-head', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'calibration': calibration()
    elif args.stage == 'resume_acceptance': resume_acceptance()
    elif args.stage == 'pretrain': pretrain(args.resume)
    elif args.stage.startswith('encode_'): encode(args.checkpoint, 'WN18RR' if args.stage == 'encode_wn' else 'cloth-copurchase')
    elif args.stage == 'probe_cloth': probe_cloth(args.checkpoint, args.resume_head)
    else:
        from came_repro import kg_affine_probe
        from came_repro.protocol_repair import candidate_cache
        from came_repro.author_loader import fingerprint
        from came_repro.atomic_io import read_json_file
        cache = candidate_cache('WN18RR', fingerprint(args.checkpoint), read_json_file(ROOT / 'configs/storage.json'))
        sys.argv = ['kg_affine_probe', '--dataset', 'WN18RR', '--checkpoint', str(args.checkpoint), '--embedding-cache', str(cache), '--graph-policy', 'train_reverse', '--seed', '0']
        if args.resume_head: sys.argv += ['--resume-head', str(args.resume_head)]
        kg_affine_probe.main()


if __name__ == '__main__': main()
