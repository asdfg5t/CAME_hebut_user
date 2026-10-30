"""Serial bounded validation-only priority pair; complete repeats remain user-operated."""
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def validation_passes(candidate, reference, metric, min_gain_pp=.5):
    values = candidate[metric], reference, min_gain_pp
    if not all(math.isfinite(v) for v in values) or min_gain_pp < 0:
        raise ValueError('Finite validation-only gate required')
    return 100 * (candidate[metric] - reference) >= min_gain_pp


def main():
    import numpy as np
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import run_child, best_resume
    from came_repro.all_unmet_repair_pipeline import collect_protocol_artifacts
    from audit_two_lp_confirmation import audit_confirmation
    profile_path = ROOT / 'configs/priority_cloth_wn_20261016.json'
    profile = read_json_file(profile_path); verify_originals()
    preflight = read_json_file(ROOT / 'artifacts/audit/priority_cloth_wn_preflight_20261016.json')
    if not preflight['passed'] or preflight['plan_sha256'] != fingerprint(profile_path): raise ValueError('Matching preflight required')
    backend = ROOT / 'scripts/priority_refinement_backend.py'
    sources = {str(p.resolve()): fingerprint(p) for p in (ROOT / 'src/came_repro').glob('*.py')}
    for p in (Path(__file__), backend, ROOT / 'scripts/run_lp_hard_negative_candidate.py', ROOT / 'scripts/audit_two_lp_confirmation.py', ROOT / 'configs/storage.json'):
        sources[str(p.resolve())] = fingerprint(p)
    identity = {'sources': sources, 'profile_sha256': fingerprint(profile_path)}
    path = ROOT / 'artifacts/workflow_state/priority_cloth_wn_current.json'
    pause = ROOT / 'artifacts/workflow_state/priority_cloth_wn_pause.json'
    guard = acquire_gpu_lock(ROOT / '.runtime/priority_cloth_wn.lock')
    ledger = read_json_file(path) if path.exists() else {'identity': identity, 'state': 'prepared', 'records': {}, 'results': {}, 'active_seconds': 0., 'formal_repeats_started': 0}
    if ledger['identity'] != identity: guard.close(); raise ValueError('Priority frozen identity changed')
    directory = ROOT / 'artifacts/remaining_main_pilots/priority_cloth_wn_20261016'; directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('priority_cloth_wn'); logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / ('pipeline_' + __import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.log'), encoding='utf-8')):
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s')); logger.addHandler(handler)
    def verify():
        if fingerprint(profile_path) != identity['profile_sha256']: raise ValueError('Frozen priority profile changed')
        for name, digest in sources.items():
            if fingerprint(name) != digest: raise ValueError('Frozen priority source changed: ' + name)
        if pause.exists(): raise RuntimeError('Priority pause retained')
    def execute(job, args, resume_name=None):
        verify()
        record = ledger['records'].setdefault(job, {'state': 'ready', 'attempts': []})
        if record['state'] == 'completed':
            for name, digest in record['artifact_sha256'].items():
                if fingerprint(name) != digest: raise ValueError('Completed result changed')
            return record
        if resume_name:
            parent = best_resume(record, resume_name)
            if parent: args += ['--resume-head', str(parent)]
        else:
            for attempt in record['attempts']:
                prior = attempt.get('run_directory')
                if prior and any((Path(prior) / n).exists() for n in ('test_predictions.npz', 'test_link_ranks.npz')):
                    raise ValueError('Interrupted test output exists; independently audit saved results before retry')
        ledger['state'] = 'running_' + job
        code, run = run_child(args, pause, ledger, path, record, logger, float('inf'))
        if code or not run:
            ledger['state'] = 'paused' if code == 75 else 'failed'
            raise RuntimeError('Priority child stopped: ' + job)
        hashes = collect_protocol_artifacts(run)
        location = Path(run) / 'checkpoints/location.json'
        if location.exists():
            last = Path(read_json_file(location)['checkpoint_directory']) / 'priority_head_last.pt'
            if last.exists(): hashes[str(last)] = fingerprint(last)
        record.update(state='completed', run_directory=run, summary=read_json_file(Path(run) / 'summary.json'), artifact_sha256=hashes)
        atomic_write_json(path, ledger); return record
    try:
        verify(); probe = acquire_gpu_lock(); probe.close()
        accepted_path = ROOT / 'artifacts/audit/priority_cloth_wn_CUDA_acceptance_20261016.json'
        accepted = read_json_file(accepted_path) if accepted_path.exists() else {}
        if not accepted.get('passed') or accepted.get('script_sha256') != fingerprint(backend) or accepted.get('profile_sha256') != fingerprint(profile_path):
            ledger['state'] = 'actual_CUDA_resume_acceptance'; atomic_write_json(path, ledger)
            env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CAME_SUITE_PARENT_PID': str(os.getpid()), 'CAME_PAUSE_FILE': str(pause), 'PYTHONIOENCODING': 'utf-8'}
            result = subprocess.run([sys.executable, '-u', '-m', 'came_repro.suite_job', '--', str(backend), '--task', 'acceptance'], cwd=ROOT, env=env)
            if result.returncode: raise RuntimeError('Actual CUDA/resume acceptance failed')
        accepted = read_json_file(accepted_path)
        for task in ('wn', 'cloth'):
            if not accepted['tasks'][task]['seed1_optimizer_scheduler_history_weights_bitexact']: raise ValueError('Incomplete acceptance')
        ledger['acceptance_sha256'] = fingerprint(accepted_path)
        for task in profile['candidate_order']:
            selected = execute('validate_' + task, [str(backend), '--task', task], 'priority_head_last.pt')
            key = 'validation_mrr' if task == 'cloth' else 'validation_accuracy'
            item = profile[task]
            gain = 100 * (selected['summary'][key] - item['reference_validation'])
            if not validation_passes(selected['summary'], item['reference_validation'], key, profile['minimum_validation_gain_pp']):
                ledger.setdefault('rejections', {})[task] = {'validation_gain_pp': gain, 'new_test_evaluations': 0, 'previous_confirmed_result_retained': True}
                atomic_write_json(path, ledger); continue
            run = Path(selected['run_directory']); summary = selected['summary']; cache = Path(summary['embedding_cache'])
            classifier = Path(read_json_file(run / 'checkpoints/location.json')['checkpoint_directory']) / ('link_head_best.pt' if task == 'cloth' else 'probe_best_raw_linear.pt')
            lock = {'dataset': summary['dataset'], 'encoder_sha256': summary['encoder_sha256'], 'classifier_checkpoint': str(classifier),
                    'classifier_sha256': fingerprint(classifier), 'validation_run_directory': str(run),
                    'validation_summary_sha256': fingerprint(run / 'summary.json'), 'embedding_cache': str(cache),
                    'inference_state_sha256': fingerprint(cache / 'inference_state.json'), 'locked_before_test': True,
                    'test_used_for_selection': False, 'graph_policy': summary['graph_policy']}
            if task == 'cloth': lock.update(head_variant=summary['head_variant'], selected_validation_mrr=summary[key])
            else: lock.update(feature_transform='train_pair_affine', selected_validation_accuracy=summary[key])
            lock_path = directory / (task + '_selection_lock.json')
            if lock_path.exists() and read_json_file(lock_path) != lock: raise ValueError('Saved validation selection lock changed')
            atomic_write_json(lock_path, lock)
            args = ['-m', 'came_repro.protocol_repair', '--stage', 'confirm_lp', '--selection-lock', str(lock_path)] if task == 'cloth' else ['-m', 'came_repro.kg_selected_confirmation', '--selection-lock', str(lock_path)]
            confirmed = execute('confirm_' + task, args)
            if task == 'cloth': report = audit_confirmation(Path(confirmed['run_directory']), lock_path)
            else:
                from came_repro.kg_graph_probe import kg_input
                final = Path(confirmed['run_directory'])
                matrix = np.load(cache / 'fused_embeddings.npy', mmap_mode='r')
                official = kg_input('WN18RR', matrix, 'test')
                with np.load(final / 'test_predictions.npz') as predictions:
                    if not np.array_equal(predictions['pairs'], official.pairs) or not np.array_equal(predictions['labels'], official.labels): raise ValueError('Full official WN test mismatch')
                    accuracy = float(np.mean(predictions['prediction'] == predictions['labels']))
                if accuracy != confirmed['summary']['test_accuracy'] or len(official) != 3134: raise ValueError('Independent WN accuracy mismatch')
                report = {'test_accuracy_pct': accuracy * 100, 'paper_accuracy_pct': 81.97, 'gap_pp': accuracy * 100 - 81.97,
                          'within_two_pp': abs(accuracy * 100 - 81.97) <= 2, 'full_official_pairs_labels_verified': True,
                          'test_evaluations': 1, 'run_directory': confirmed['run_directory'], 'PNG_visual_verified': False}
            report['validation_gain_pp'] = gain
            atomic_write_json(directory / (task + '_independent_audit.json'), report)
            ledger['results'][task] = report; atomic_write_json(path, ledger)
        ledger.update(state='completed_bounded_priority_diagnostics_pending_manual_delivery_review',
                      all_targets_met=all(ledger['results'].get(t, {}).get('within_two_pp', False) for t in ('wn', 'cloth')),
                      formal_repeats_started=0, Products_old_pause_retained=True)
        atomic_write_json(path, ledger)
    except BaseException as error:
        ledger.update(state='paused' if pause.exists() or ledger.get('state') == 'paused' else 'failed', error=str(error)); atomic_write_json(path, ledger); raise
    finally: guard.close()


if __name__ == '__main__': main()
