"""User-run entry: repeat only fully verified LP deliveries with their exact negative policy."""
import argparse
import datetime
import logging
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
BACKENDS = {
    'uniform_one': ('scripts/run_locked_lp_repeats.py', 'artifacts/audit/locked_lp_repeat_CUDA_acceptance_20261014.json'),
    'hard_four': ('scripts/run_lp_hard_negative_candidate.py', 'artifacts/audit/hard_lp_CUDA_acceptance_20261014.json'),
}


def validate_training_policy(kind, summary):
    """Refuse using a different training policy than the delivered seed0."""
    if kind not in BACKENDS:
        raise ValueError('Unknown delivered LP training policy')
    hard = summary.get('negative_pool_size') == 4
    if (kind == 'hard_four') != hard:
        raise ValueError('Delivered negative policy and repeat backend differ')
    if hard and summary.get('negative_policy') != 'max_score_train_and_reverse_known_nonedge':
        raise ValueError('Hard-negative known-positive filter changed')
    if not hard and summary.get('negative_pool_size', 1) != 1:
        raise ValueError('Uniform-one delivery has an unexpected negative pool')


def child_arguments(kind, dataset, seed, reference):
    if seed not in range(1, 10):
        raise ValueError('Repeat only the fixed new seeds1-9; seed0 is retained')
    script = BACKENDS[kind][0]
    return [script] + (['--train-one'] if kind == 'uniform_one' else []) + [
        '--dataset', dataset, '--seed', str(seed), '--reference-run', str(reference)]


def validate_delivery(delivery):
    from came_repro.atomic_io import read_json_file
    from came_repro.author_loader import fingerprint
    if delivery.get('schema') != 'verified_two_lp_delivery_v1':
        raise ValueError('An independently accepted delivery manifest is required')
    verified = {}
    for dataset, item in delivery['tasks'].items():
        if dataset not in ('sports-copurchase', 'cloth-copurchase'):
            raise ValueError('Delivery contains an unrelated task')
        for name, digest in item['artifact_sha256'].items():
            if fingerprint(name) != digest:
                raise ValueError('Delivered artifact changed: ' + name)
        validation = Path(item['validation_run']).resolve()
        confirmation = Path(item['confirmation_run']).resolve()
        audit_path = Path(item['independent_audit']).resolve()
        required = {str(validation / 'summary.json'), str(confirmation / 'summary.json'), str(audit_path)}
        bound = {str(Path(p).resolve()) for p in item['artifact_sha256']}
        if not required.issubset(bound):
            raise ValueError('Delivery must bind both summaries and independent audit')
        summary = read_json_file(validation / 'summary.json')
        confirmed = read_json_file(confirmation / 'summary.json')
        audit = read_json_file(audit_path)
        validate_training_policy(item['training_kind'], summary)
        if summary['dataset'] != dataset or summary['seed'] != 0 or summary['test_evaluations'] != 0:
            raise ValueError('Delivery reference is not a held-out seed0 validation run')
        if Path(confirmed['validation_run_directory']).resolve() != validation or confirmed['test_evaluations'] != 1:
            raise ValueError('Confirmation does not belong to the delivered validation run')
        if Path(audit['confirmation_run']).resolve() != confirmation or not audit['figure_visual_verified']:
            raise ValueError('Independent confirmation and visual audit are required')
        paper = 37.68 if dataset == 'sports-copurchase' else 28.88
        value = confirmed['test']['official']['mrr'] * 100
        if abs(value - paper) > 2 or abs(value - audit['test_mrr_pct']) > 2e-5:
            raise ValueError('Delivered full-test result does not meet the stated target')
        backend, acceptance_path = BACKENDS[item['training_kind']]
        accepted = read_json_file(ROOT / acceptance_path)
        if not accepted['passed'] or not accepted['actual_Torch_DGL_CUDA'] or accepted['script_sha256'] != fingerprint(ROOT / backend):
            raise ValueError('Exact repeat backend requires actual CUDA acceptance')
        if not accepted.get('seed1_pause_resume_bitexact_weights_history'):
            raise ValueError('Nonzero seed pause/resume acceptance is required before delivery')
        if item['training_kind'] == 'hard_four' and accepted['profile_sha256'] != fingerprint(ROOT / 'configs/lp_negative_objective_20261014.json'):
            raise ValueError('Accepted hard-negative profile changed')
        verified[dataset] = (item, validation, confirmed)
    return verified


def main():
    import numpy as np
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--delivery', type=Path, default=ROOT / 'artifacts/remaining_main_pilots/two_lp_priority/verified_delivery.json')
    args = parser.parse_args()
    pointer = (args.delivery if args.delivery.is_absolute() else ROOT / args.delivery).resolve()
    if not pointer.exists():
        print('NOT_READY: no fully verified delivery; GPU jobs=0.', flush=True)
        return
    verified = validate_delivery(read_json_file(pointer))
    if not verified:
        print('NOT_READY: no accepted task; GPU jobs=0.', flush=True)
        return
    if args.dry_run:
        print('ACCEPTED_TASKS=' + ','.join(verified) + '; reuse seed0; new seeds1-9; GPU jobs=0.', flush=True)
        return
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child, best_resume
    from came_repro.all_unmet_repair_pipeline import collect_protocol_artifacts
    from came_repro.storage_budget import check_budget
    from audit_two_lp_confirmation import audit_confirmation
    try:
        probe = acquire_gpu_lock()
        probe.close()
    except RuntimeError:
        print('GPU_BUSY: diagnostics are still running; no repeats started.', flush=True)
        return
    workflow = 'verified_two_lp_repeats' if pointer.name == 'verified_delivery.json' else pointer.stem + '_repeats'
    guard = acquire_gpu_lock(ROOT / '.runtime' / (workflow + '.lock'))
    ledger_path = ROOT / 'artifacts/workflow_state' / (workflow + '_current.json')
    pause = ROOT / 'artifacts/workflow_state' / (workflow + '_pause.json')
    directory = ROOT / 'artifacts' / workflow
    directory.mkdir(parents=True, exist_ok=True)
    frozen_paths = list((ROOT / 'src/came_repro').glob('*.py')) + [Path(__file__), ROOT / 'scripts/audit_two_lp_confirmation.py',
        ROOT / 'scripts/run_locked_lp_repeats.py', ROOT / 'scripts/run_lp_hard_negative_candidate.py',
        ROOT / 'configs/lp_negative_objective_20261014.json', pointer]
    identity = {str(p.resolve()): fingerprint(p) for p in frozen_paths}
    logger = logging.getLogger('verified_two_lp_repeats')
    logger.setLevel(logging.INFO)
    for handler in [logging.StreamHandler(), logging.FileHandler(directory / ('pipeline_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.log'), encoding='utf-8')]:
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    try:
        ledger = read_json_file(ledger_path) if ledger_path.exists() else {'identity': identity, 'state': 'ready', 'records': {}, 'active_seconds': 0., 'statistics': {}}
        if ledger['identity'] != identity:
            raise ValueError('Repeat identity changed; audit before resuming')
        verify_originals()
        check_budget(additional_cache_bytes=2**30)

        def execute(job, arguments, resume_name=None):
            for name, digest in identity.items():
                if fingerprint(name) != digest:
                    raise ValueError('Frozen repeat identity changed: ' + name)
            rec = ledger['records'].setdefault(job, {'state': 'ready', 'attempts': []})
            if rec['state'] == 'completed':
                for name, digest in rec['artifact_sha256'].items():
                    if fingerprint(name) != digest:
                        raise ValueError('Completed repeat artifact changed')
                return rec
            if pause.exists():
                ledger['state'] = 'paused'
                atomic_write_json(ledger_path, ledger)
                raise RuntimeError('Repeat pause retained; no child launched')
            if resume_name:
                checkpoint = best_resume(rec, resume_name)
                if checkpoint:
                    arguments += ['--resume-head', str(checkpoint)]
            else:
                for attempt in rec['attempts']:
                    run = attempt.get('run_directory')
                    if run and (Path(run) / 'test_link_ranks.npz').exists():
                        raise ValueError('Interrupted test artifacts exist; audit saved result before any retry')
            ledger['state'] = 'running'
            code, run = run_child(arguments, pause, ledger, ledger_path, rec, logger, float('inf'))
            if code or not run:
                ledger['state'] = rec['state'] = 'paused' if code == 75 else 'failed'
                atomic_write_json(ledger_path, ledger)
                raise RuntimeError('Repeat child stopped: ' + job)
            rec.update(state='completed', run_directory=run, summary=read_json_file(Path(run) / 'summary.json'), artifact_sha256=collect_protocol_artifacts(run))
            atomic_write_json(ledger_path, ledger)
            return rec

        for dataset, (item, reference, confirmed0) in verified.items():
            scores = [confirmed0['test']['official']['mrr'] * 100]
            for seed in range(1, 10):
                logger.info('MAIN_TASK %s dataset=%s seed=%d/9; frozen %s', item['main_task_number'], dataset, seed, item['training_kind'])
                rec = execute(dataset + '_validation_s' + str(seed), child_arguments(item['training_kind'], dataset, seed, reference), 'link_head_last.pt')
                run = Path(rec['run_directory'])
                summary = rec['summary']
                validate_training_policy(item['training_kind'], summary)
                cache = Path(summary['embedding_cache'])
                best = Path(read_json_file(run / 'checkpoints/location.json')['checkpoint_directory']) / 'link_head_best.pt'
                lock = {'dataset': dataset, 'encoder_sha256': summary['encoder_sha256'], 'classifier_checkpoint': str(best), 'classifier_sha256': fingerprint(best),
                        'validation_run_directory': str(run), 'validation_summary_sha256': fingerprint(run / 'summary.json'),
                        'embedding_cache': str(cache), 'inference_state_sha256': fingerprint(cache / 'inference_state.json'),
                        'locked_before_test': True, 'test_used_for_selection': False, 'selected_validation_mrr': summary['validation_mrr'],
                        'head_variant': summary['head_variant'], 'graph_policy': 'upstream_train_bidirected_unique'}
                lock_path = directory / (dataset + '_s' + str(seed) + '_selection_lock.json')
                if lock_path.exists() and read_json_file(lock_path) != lock:
                    raise ValueError('Selected repeat head changed')
                atomic_write_json(lock_path, lock)
                confirmed = execute(dataset + '_confirmation_s' + str(seed), ['-m', 'came_repro.protocol_repair', '--stage', 'confirm_lp', '--selection-lock', str(lock_path)])
                audit = audit_confirmation(Path(confirmed['run_directory']), lock_path)
                atomic_write_json(directory / (dataset + '_s' + str(seed) + '_independent_audit.json'), audit)
                scores.append(audit['test_mrr_pct'])
                paper = 37.68 if dataset == 'sports-copurchase' else 28.88
                ledger['statistics'][dataset] = {'completed': len(scores), 'expected': 10, 'scores_pct': scores,
                    'mean_mrr_pct': float(np.mean(scores)), 'sample_sd_pp': float(np.std(scores, ddof=1)), 'paper_mrr_pct': paper,
                    'gap_pp': float(np.mean(scores)) - paper, 'all_seeds_included': True, 'complete_repeat_grid': len(scores) == 10}
                atomic_write_json(ledger_path, ledger)
        ledger['state'] = 'completed_delivered_two_lp_repeats'
        atomic_write_json(ledger_path, ledger)
    finally:
        guard.close()


if __name__ == '__main__':
    main()
