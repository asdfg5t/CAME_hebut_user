"""Wait for user Sports repeats, then run bounded Cloth/WN/Products diagnostics serially.

No baselines, few-shot, ablations, formal repeat training, or Books score manipulation.
"""
import datetime
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sports_ready(ledger):
    stats = ledger.get('statistics', {}).get('sports-copurchase', {})
    return (ledger.get('state') == 'completed_delivered_two_lp_repeats'
            and stats.get('completed') == 10 and stats.get('complete_repeat_grid') is True)


def main():
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child, best_resume
    from came_repro.all_unmet_repair_pipeline import collect_protocol_artifacts
    from came_repro.storage_budget import check_budget
    from audit_two_lp_confirmation import audit_confirmation
    plan_path = ROOT / 'configs/post_sports_main_repairs_20261015.json'
    plan = read_json_file(plan_path)
    sources = {str(p.resolve()): fingerprint(p) for p in (ROOT / 'src/came_repro').glob('*.py')}
    for name in plan['frozen_infrastructure']:
        sources[str((ROOT / name).resolve())] = fingerprint(ROOT / name)
    sources[str(Path(__file__).resolve())] = fingerprint(__file__)
    identity = {'sources': sources, 'plan_sha256': fingerprint(plan_path)}
    path = ROOT / 'artifacts/workflow_state/post_sports_main_repairs_current.json'
    pause = ROOT / 'artifacts/workflow_state/post_sports_main_repairs_pause.json'
    sports_path = ROOT / 'artifacts/workflow_state/verified_job07_sports_delivery_repeats_current.json'
    ledger = read_json_file(path) if path.exists() else {
        'identity': identity, 'state': 'waiting_for_user_Sports_ten_seeds', 'records': {},
        'active_seconds': 0., 'formal_repeat_jobs_started': 0, 'verified_results': {}}
    if ledger['identity'] != identity:
        raise ValueError('Post-Sports frozen identity changed; audit before continuation')
    guard = acquire_gpu_lock(ROOT / '.runtime/post_sports_main_repairs.lock')
    atomic_write_json(path, ledger)
    directory = ROOT / 'artifacts/remaining_main_pilots/post_sports_main_repairs'
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('post_sports_main_repairs')
    logger.setLevel(logging.INFO)
    for handler in [logging.StreamHandler(), logging.FileHandler(directory / ('pipeline_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.log'), encoding='utf-8')]:
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    try:
        logger.info('WAITING: preserve user Sports training; diagnostics require all ten seeds and free global GPU lock')
        while True:
            if pause.exists():
                ledger['state'] = 'paused'
                atomic_write_json(path, ledger)
                return
            sports = read_json_file(sports_path)
            if sports['state'] in ('failed', 'paused'):
                ledger['state'] = 'deferred_user_Sports_failure_or_pause'
                atomic_write_json(path, ledger)
                return
            if sports_ready(sports):
                try:
                    probe = acquire_gpu_lock()
                    probe.close()
                    break
                except RuntimeError:
                    pass
            time.sleep(15)

        def verify_identity():
            if fingerprint(plan_path) != identity['plan_sha256']:
                raise ValueError('Frozen post-Sports plan changed')
            for name, digest in sources.items():
                if fingerprint(name) != digest:
                    raise ValueError('Frozen post-Sports source changed: ' + name)
            verify_originals()
        verify_identity()
        check_budget(additional_cache_bytes=plan['head_metadata_reservation_bytes'])
        ledger['Sports_ten_seeds_completed_before_diagnostics'] = True
        atomic_write_json(path, ledger)

        def execute(job, arguments, resume_name=None, initial_resume=None):
            verify_identity()
            rec = ledger['records'].setdefault(job, {'state': 'ready', 'attempts': []})
            if rec['state'] == 'completed':
                for name, digest in rec['artifact_sha256'].items():
                    if fingerprint(name) != digest:
                        raise ValueError('Completed diagnostic artifact changed')
                return rec
            if pause.exists():
                ledger['state'] = 'paused'
                atomic_write_json(path, ledger)
                raise RuntimeError('User diagnostic pause retained')
            if resume_name:
                checkpoint = best_resume(rec, resume_name) or initial_resume
                if checkpoint:
                    arguments += ['--resume-head', str(checkpoint)]
            else:
                for attempt in rec['attempts']:
                    prior = attempt.get('run_directory')
                    if prior and any((Path(prior) / filename).exists() for filename in ('test_link_ranks.npz', 'test_predictions.npz')):
                        raise ValueError('Interrupted test output exists; audit saved result, do not blindly retest')
            ledger['state'] = 'running_' + job
            code, run = run_child(arguments, pause, ledger, path, rec, logger, float('inf'))
            if code or not run:
                ledger['state'] = rec['state'] = 'paused' if code == 75 else 'failed'
                atomic_write_json(path, ledger)
                raise RuntimeError('Diagnostic child stopped: ' + job)
            rec.update(state='completed', run_directory=run, summary=read_json_file(Path(run) / 'summary.json'),
                       artifact_sha256=collect_protocol_artifacts(run))
            atomic_write_json(path, ledger)
            return rec

        # Exactly one predeclared predictor candidate; no test-dependent parameter search.
        accepted_path = ROOT / 'artifacts/audit/cloth_predictor_CUDA_acceptance_20261015.json'
        accepted = read_json_file(accepted_path) if accepted_path.exists() else {}
        candidate = ROOT / 'scripts/run_cloth_predictor_candidate.py'
        cloth_profile = ROOT / 'configs/cloth_predictor_structure_20261015.json'
        if not accepted.get('passed') or accepted.get('script_sha256') != fingerprint(candidate) or accepted.get('profile_sha256') != fingerprint(cloth_profile):
            verify_identity()
            env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CAME_PAUSE_FILE': str(pause),
                   'CAME_SUITE_PARENT_PID': str(os.getpid()), 'PYTHONIOENCODING': 'utf-8'}
            code = subprocess.run([sys.executable, '-u', '-m', 'came_repro.suite_job', '--', str(candidate), '--acceptance'], cwd=ROOT, env=env).returncode
            if code:
                raise RuntimeError('Cloth exact predictor CUDA/resume acceptance failed')
        accepted = read_json_file(accepted_path)
        if not accepted.get('passed') or not accepted['seed1_complete_optimizer_RNG_resume_bitexact']:
            raise ValueError('Cloth actual acceptance incomplete')
        ledger['cloth_actual_CUDA_acceptance_sha256'] = fingerprint(accepted_path)
        atomic_write_json(path, ledger)
        selected = execute('validate_cloth_predictor', [str(candidate)], 'link_head_last.pt')
        profile = read_json_file(cloth_profile)
        gain = 100 * (selected['summary']['validation_mrr'] - profile['reference_validation_mrr'])
        if gain >= profile['minimum_validation_gain_pp']:
            run = Path(selected['run_directory'])
            summary = selected['summary']
            cache = Path(summary['embedding_cache'])
            best = Path(read_json_file(run / 'checkpoints/location.json')['checkpoint_directory']) / 'link_head_best.pt'
            lock = {'dataset': 'cloth-copurchase', 'encoder_sha256': summary['encoder_sha256'],
                    'classifier_checkpoint': str(best), 'classifier_sha256': fingerprint(best),
                    'validation_run_directory': str(run), 'validation_summary_sha256': fingerprint(run / 'summary.json'),
                    'embedding_cache': str(cache), 'inference_state_sha256': fingerprint(cache / 'inference_state.json'),
                    'locked_before_test': True, 'test_used_for_selection': False,
                    'selected_validation_mrr': summary['validation_mrr'], 'head_variant': summary['head_variant'],
                    'graph_policy': summary['graph_policy']}
            lock_path = directory / 'cloth_predictor_selection_lock.json'
            if lock_path.exists() and read_json_file(lock_path) != lock:
                raise ValueError('Cloth selected head lock changed')
            atomic_write_json(lock_path, lock)
            confirmed = execute('confirm_cloth_predictor', ['-m', 'came_repro.protocol_repair', '--stage', 'confirm_lp', '--selection-lock', str(lock_path)])
            audit = audit_confirmation(Path(confirmed['run_directory']), lock_path)
            audit['validation_gain_pp'] = gain
            atomic_write_json(directory / 'cloth_predictor_independent_audit.json', audit)
            ledger['verified_results']['cloth-copurchase'] = audit
        else:
            ledger.setdefault('validation_rejections', {})['cloth-copurchase'] = {'gain_pp': gain, 'new_test_evaluations': 0}
        atomic_write_json(path, ledger)

        # Resume only the unfinished original WN candidate, never its obsolete LP queue.
        wn_reference = ROOT / plan['WN']['reference_validation_run']
        if fingerprint(wn_reference / 'summary.json') != plan['WN']['reference_validation_summary_sha256']:
            raise ValueError('WN validation reference changed')
        cp = ROOT / plan['WN']['checkpoint_location_file']
        parent = Path(read_json_file(cp)['checkpoint_directory']) / 'probe_last.pt'
        if fingerprint(parent) != plan['WN']['parent_checkpoint_sha256']:
            raise ValueError('Original paused WN checkpoint changed')
        state = torch.load(str(parent), map_location='cpu', weights_only=False)
        if state['schema'] != 'came_wolfe_linear_v1' or state['next_epoch'] != state['completed_epochs'] + 1 or len(state['history_rows']) != state['completed_epochs']:
            raise ValueError('Incomplete WN optimizer/RNG/history checkpoint')
        if state['config'] != read_json_file(cp.parent.parent / 'config.json'):
            raise ValueError('Original WN config changed')
        atomic_write_json(directory / 'WN_original_pause_resume_audit.json', {
            'parent_checkpoint_sha256': fingerprint(parent), 'completed_steps': state['completed_epochs'],
            'next_step': state['next_epoch'], 'max_steps_unchanged': 500,
            'legacy_convergence_pause_preserved': True, 'private_current_pause': str(pause),
            'validation_only_same_frozen_solver': True})
        wn = execute('validate_WN_continuation', ['scripts/run_convergence_candidate.py', '--stage', 'wn', '--dataset', 'WN18RR', '--reference-run', str(wn_reference)], 'probe_last.pt', parent)
        gain = 100 * (wn['summary']['validation_accuracy'] - read_json_file(wn_reference / 'summary.json')['validation_accuracy'])
        if gain >= plan['WN']['minimum_validation_gain_pp']:
            run = Path(wn['run_directory']); summary = wn['summary']; cache = Path(summary['embedding_cache'])
            best = Path(read_json_file(run / 'checkpoints/location.json')['checkpoint_directory']) / 'probe_best_raw_linear.pt'
            lock = {'dataset': 'WN18RR', 'encoder_sha256': summary['encoder_sha256'], 'classifier_checkpoint': str(best),
                    'classifier_sha256': fingerprint(best), 'validation_run_directory': str(run),
                    'validation_summary_sha256': fingerprint(run / 'summary.json'), 'embedding_cache': str(cache),
                    'inference_state_sha256': fingerprint(cache / 'inference_state.json'), 'locked_before_test': True,
                    'test_used_for_selection': False, 'selected_validation_accuracy': summary['validation_accuracy'],
                    'graph_policy': summary['graph_policy'], 'feature_transform': 'train_pair_affine'}
            lock_path = directory / 'WN_selection_lock.json'
            if lock_path.exists() and read_json_file(lock_path) != lock:
                raise ValueError('WN saved head lock changed')
            atomic_write_json(lock_path, lock)
            confirmed = execute('confirm_WN_continuation', ['-m', 'came_repro.kg_selected_confirmation', '--selection-lock', str(lock_path)])
            final = Path(confirmed['run_directory'])
            with np.load(final / 'test_predictions.npz') as pred:
                from came_repro.kg_graph_probe import kg_input
                embeddings = np.load(cache / 'fused_embeddings.npy', mmap_mode='r')
                official = kg_input('WN18RR', embeddings, 'test')
                if not np.array_equal(pred['pairs'], official.pairs) or not np.array_equal(pred['labels'], official.labels):
                    raise ValueError('WN full official pairs/labels mismatch')
                value = float(np.mean(pred['prediction'] == pred['labels']))
                if value != confirmed['summary']['test_accuracy']:
                    raise ValueError('Independent WN accuracy mismatch')
                report = {'test_accuracy_pct': value * 100, 'official_queries': len(official),
                          'paper_accuracy_pct': 81.97, 'gap_pp': value * 100 - 81.97,
                          'within_two_pp': abs(value * 100 - 81.97) <= 2, 'test_evaluations': 1,
                          'full_official_pairs_labels_verified': True, 'PNG_visual_verified': False}
            atomic_write_json(directory / 'WN_independent_audit.json', report)
            ledger['verified_results']['WN18RR'] = report
        else:
            ledger.setdefault('validation_rejections', {})['WN18RR'] = {'gain_pp': gain, 'new_test_evaluations': 0}
        atomic_write_json(path, ledger)

        # Products owns the GPU lock itself; do not wrap it in suite_job a second time.
        verify_identity()
        product_path = ROOT / 'artifacts/workflow_state/products_single_seed_current.json'
        existing = read_json_file(product_path) if product_path.exists() else {}
        if existing.get('state') != 'completed_verified_official_single_seed':
            if existing.get('state') == 'full_official_test':
                raise ValueError('Interrupted Products test needs saved-output audit before any retry')
            arguments = [sys.executable, '-u', 'scripts/run_products_single_validation.py', '--after-sports', '--pause-file', str(pause)]
            for attempt in reversed(existing.get('attempts', [])):
                location = Path(attempt['run_directory']) / 'checkpoints/location.json'
                if location.exists():
                    saved = Path(read_json_file(location)['checkpoint_directory']) / 'probe_last.pt'
                    if saved.exists():
                        arguments += ['--resume-head', str(saved)]
                        break
            ledger['state'] = 'running_full_Products_single_seed'
            atomic_write_json(path, ledger)
            code = subprocess.run(arguments, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'),
                'CAME_SUITE_PARENT_PID': str(os.getpid()), 'PYTHONIOENCODING': 'utf-8'}).returncode
            if code:
                raise RuntimeError('Products single-seed stopped; retain its cursor/checkpoint')
        products = read_json_file(product_path)
        if products['state'] != 'completed_verified_official_single_seed':
            raise ValueError('Products full official single-seed not completed')
        ledger['verified_results']['ogbn-products'] = {
            'test_accuracy_pct': products['summary']['test_accuracy'] * 100, 'paper_accuracy_pct': 81.84,
            'gap_pp': products['summary']['test_accuracy'] * 100 - 81.84,
            'within_two_pp': abs(products['summary']['test_accuracy'] * 100 - 81.84) <= 2,
            'full_official_test_nodes': 2213091, 'PNG_visual_verified': False,
            'run_directory': products['run_directory']}
        ledger.update(state='completed_bounded_main_diagnostics_Books_protocol_unresolved',
                      Books_action='Preserve verified high MRR; exact author/publisher protocol remains unresolved, no invented negatives or artificial degradation',
                      all_four_targets_met=False, formal_repeat_jobs_started=0)
        atomic_write_json(path, ledger)
    except BaseException as error:
        ledger.update(state='paused' if pause.exists() or ledger.get('state') == 'paused' else 'failed', error=str(error))
        atomic_write_json(path, ledger)
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    main()
