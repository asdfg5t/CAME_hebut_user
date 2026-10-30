"""After bounded diagnostics, audit eligible results and prepare a human visual-review handoff.

Never start formal repeats and never certify a figure that an agent has not inspected.
"""
import json
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


def main():
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from audit_two_lp_confirmation import audit_confirmation
    from run_verified_two_lp_repeats import BACKENDS, validate_training_policy
    path = ROOT / 'artifacts/workflow_state/two_lp_handoff_current.json'
    pause = ROOT / 'artifacts/workflow_state/two_lp_handoff_pause.json'
    files = list((ROOT / 'src/came_repro').glob('*.py')) + [Path(__file__), ROOT / 'scripts/audit_two_lp_confirmation.py',
        ROOT / 'scripts/run_verified_two_lp_repeats.py', ROOT / 'scripts/run_locked_lp_repeats.py',
        ROOT / 'scripts/run_lp_hard_negative_candidate.py', ROOT / 'configs/lp_negative_objective_20261014.json']
    identity = {str(p.resolve()): fingerprint(p) for p in files}
    ledger = read_json_file(path) if path.exists() else {'identity': identity, 'state': 'waiting_for_bounded_diagnostics', 'new_formal_repeat_jobs': 0}
    if ledger['identity'] != identity:
        raise ValueError('Handoff identity changed; audit before continuing')
    guard = acquire_gpu_lock(ROOT / '.runtime/two_lp_handoff.lock')
    atomic_write_json(path, ledger)
    try:
        while True:
            if pause.exists():
                ledger['state'] = 'paused'
                atomic_write_json(path, ledger)
                return
            first = read_json_file(ROOT / 'artifacts/workflow_state/two_lp_priority_current.json')
            hard = read_json_file(ROOT / 'artifacts/workflow_state/two_lp_hard_negative_current.json')
            if any(s['state'] in ('failed', 'paused') or s['state'].startswith('deferred_') for s in (first, hard)):
                ledger['state'] = 'deferred_diagnostic_failure_or_pause'
                atomic_write_json(path, ledger)
                return
            if first['state'] == 'completed_two_lp_bounded_diagnostics' and hard['state'] in ('completed_bounded_hard_negative_diagnostics', 'completed_no_failed_tasks_to_repair'):
                try:
                    probe = acquire_gpu_lock()
                    probe.close()
                    break
                except RuntimeError:
                    pass
            time.sleep(15)
        for name, digest in identity.items():
            if fingerprint(name) != digest:
                raise ValueError('Frozen handoff source changed: ' + name)
        verify_originals()
        tasks = {}
        directory = ROOT / 'artifacts/remaining_main_pilots/two_lp_priority'
        directory.mkdir(parents=True, exist_ok=True)
        for dataset in ('sports-copurchase', 'cloth-copurchase'):
            # A hard candidate is considered only after the declared validation gate.
            # Never choose between two tested models using the better test score.
            if 'confirm_' + dataset in hard['records']:
                record = hard['records']['confirm_' + dataset]
                kind = 'hard_four'
                lock_path = ROOT / 'artifacts/remaining_main_pilots/hard_lp' / (dataset + '_selection_lock.json')
            elif 'confirm_' + dataset in first['records']:
                record = first['records']['confirm_' + dataset]
                kind = 'uniform_one'
                lock_path = directory / (dataset + '_selection_lock.json')
            else:
                continue
            if record['state'] != 'completed':
                raise ValueError('Candidate confirmation is incomplete')
            run = Path(record['run_directory'])
            report = audit_confirmation(run, lock_path)
            audit_path = ROOT / 'artifacts/audit' / ('two_lp_handoff_' + dataset + '_20261014.json')
            atomic_write_json(audit_path, report)
            if not report['within_two_pp']:
                ledger.setdefault('unmet_reports', {})[dataset] = report
                continue
            validation = Path(report['validation_run'])
            summary = read_json_file(validation / 'summary.json')
            validate_training_policy(kind, summary)
            backend, acceptance = BACKENDS[kind]
            accepted_path = ROOT / acceptance
            accepted = read_json_file(accepted_path) if accepted_path.exists() else {}
            if not accepted.get('passed') or accepted.get('script_sha256') != fingerprint(ROOT / backend) or not accepted.get('seed1_pause_resume_bitexact_weights_history'):
                if pause.exists():
                    raise RuntimeError('Handoff paused before CUDA acceptance')
                code = subprocess.run([sys.executable, '-u', '-m', 'came_repro.suite_job', '--', backend, '--acceptance'],
                    cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CAME_PAUSE_FILE': str(pause), 'PYTHONIOENCODING': 'utf-8'}).returncode
                if code:
                    raise RuntimeError('Repeat CUDA acceptance failed; no delivery')
            accepted = read_json_file(accepted_path)
            if not accepted['passed'] or accepted['script_sha256'] != fingerprint(ROOT / backend) or not accepted['seed1_pause_resume_bitexact_weights_history']:
                raise ValueError('Repeat implementation acceptance changed')
            lock = read_json_file(lock_path)
            bound = [validation / 'summary.json', validation / 'config.json', run / 'summary.json', run / 'manifest.json',
                     run / 'test_link_ranks.npz', audit_path, lock_path, Path(lock['classifier_checkpoint']),
                     Path(lock['embedding_cache']) / 'inference_state.json', accepted_path]
            tasks[dataset] = {'training_kind': kind, 'validation_run': str(validation), 'confirmation_run': str(run),
                'main_task_number': '07' if dataset == 'sports-copurchase' else '08',
                'independent_audit': str(audit_path), 'paper_mrr_pct': report['paper_mrr_pct'],
                'artifact_sha256': {str(p.resolve()): fingerprint(p) for p in bound}}
        atomic_write_json(directory / 'pending_delivery_visual_review.json', {
            'schema': 'verified_two_lp_delivery_v1', 'tasks': tasks,
            'not_ready_for_formal_training': True, 'requires_agent_visual_inspection_and_manifest_rebinding': True,
            'new_formal_repeat_jobs': 0})
        ledger.update(state='pending_visual_review' if tasks else 'completed_no_eligible_delivery', eligible_tasks=list(tasks),
                      entry='scripts/run_verified_two_lp_repeats.py', new_formal_repeat_jobs=0)
        atomic_write_json(path, ledger)
        print(json.dumps({'state': ledger['state'], 'eligible_tasks': list(tasks), 'new_formal_repeat_jobs': 0}))
    except BaseException as error:
        ledger.update(state='paused' if pause.exists() else 'failed', error=str(error))
        atomic_write_json(path, ledger)
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    main()
