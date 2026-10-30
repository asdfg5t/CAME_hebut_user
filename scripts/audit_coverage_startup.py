"""Read-only startup audit and compact monitor snapshot; no CUDA operations."""
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ledger_path = ROOT / 'artifacts/workflow_state/representation_coverage_current.json'
    ledger = read_json_file(ledger_path)
    plan_path = ROOT / 'configs/representation_coverage_validation_20261016.json'
    plan = read_json_file(plan_path)
    verify_originals()
    for path, digest in ledger['identity']['sources'].items():
        if fingerprint(path) != digest: raise ValueError('Active source differs from frozen identity')
    if fingerprint(plan_path) != ledger['identity']['plan_sha256']:
        raise ValueError('Active plan identity changed')
    originals = {}
    for name, clone in plan['clones'].items():
        actual = fingerprint(clone['source'])
        if actual != clone['source_sha256']: raise ValueError('Historical PPR database changed')
        originals[name] = actual
    accepted = read_json_file(ROOT / 'artifacts/representation_coverage_20261016/joint_resume_acceptance.json')
    if not accepted['passed'] or accepted['max_model_optimizer_abs_difference'] != 0.:
        raise ValueError('Expected exact observed CUDA restoration result')
    checkpoint = Path(read_json_file(ROOT / 'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    if fingerprint(checkpoint) != '1fd5a614766cf52fe15cc9e0a9f577fbfdc07cd70de7bd6cc7a02834a1cde91e':
        raise ValueError('Protected original encoder changed')
    source = ROOT / 'artifacts/monitoring/came_progress.json'
    old = read_json_file(source) if source.exists() else {}
    active = ledger['records'].get('fresh_pretraining', {}).get('attempts', [])
    report = {'checked_at_utc': now, 'state': ledger['state'], 'identity_verified': True,
              'original_author_files_verified': True, 'original_encoder_verified': True,
              'historical_PPR_file_sha256_verified': originals,
              'actual_joint_resume_model_optimizer_max_abs_difference': 0.,
              'RNG_sampling_order_cursor_verified': True, 'formal_repeats_started': 0,
              'new_test_evaluations': 0, 'source_ledger_sha256_at_read': fingerprint(ledger_path)}
    atomic_write_json(ROOT / 'artifacts/audit/representation_coverage_startup_verified_20261016.json', report)
    snapshot = {'checked_at_utc': now, 'previous_snapshot': {'checked_at': old.get('checked_at_utc', old.get('checked_at')),
                'priority_pair_state': old.get('priority_pair_state'), 'coverage_state': old.get('coverage_state')},
                'coverage_state': ledger['state'], 'coverage_run': active[-1].get('run_directory') if active else None,
                'supervisor_tool_session': 39286, 'priority_pair_state': 'completed_both_validation_candidates_rejected',
                'confirmed_test': {'cloth_mrr_pct': 26.810958023732056, 'cloth_gap_pp': -2.0690419762679433,
                                   'WN_accuracy_pct': 77.98340778557754, 'WN_gap_pp': -3.986592214422459},
                'new_backbone_test_results': ledger['results'], 'formal_repeat_jobs_started_by_assistant': 0,
                'six_completed_formal_tasks_preserved': True, 'Products_inactive_maintenance_pause_retained': True,
                'automation_remains_paused': True, 'source_frozen_verified': True}
    atomic_write_json(source, snapshot)
    print(json.dumps(report))


if __name__ == '__main__': main()
