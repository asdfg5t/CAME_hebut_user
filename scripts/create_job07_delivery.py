"""Bind the already independently audited Sports seed0 to an immutable user-run manifest."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from run_verified_two_lp_repeats import validate_delivery
    verify_originals()
    audit_path = ROOT / 'artifacts/audit/sports_hard_full_artifacts_20261014.json'
    audit = read_json_file(audit_path)
    if not audit['figure_visual_verified'] or not audit['within_two_pp']:
        raise ValueError('Independent full-test and visual acceptance are required')
    validation = Path(audit['validation_run']).resolve()
    confirmation = Path(audit['confirmation_run']).resolve()
    lock_path = ROOT / 'artifacts/remaining_main_pilots/hard_lp/sports-copurchase_selection_lock.json'
    lock = read_json_file(lock_path)
    if fingerprint(validation / 'summary.json') != lock['validation_summary_sha256']:
        raise ValueError('Audited validation summary changed')
    if fingerprint(lock['classifier_checkpoint']) != lock['classifier_sha256']:
        raise ValueError('Audited best head changed')
    figure = Path(audit['figure'])
    if fingerprint(figure) != audit['figure_sha256']:
        raise ValueError('Visually inspected figure changed')
    acceptance = ROOT / 'artifacts/audit/hard_lp_CUDA_acceptance_20261014.json'
    state = Path(lock['embedding_cache']) / 'inference_state.json'
    if fingerprint(state) != lock['inference_state_sha256']:
        raise ValueError('Frozen embedding identity changed')
    bound = [audit_path, validation / 'summary.json', validation / 'config.json',
             validation / 'logs/metrics.jsonl', validation / 'logs/metrics.csv', validation / 'logs/train.log',
             validation / 'validation_link_ranks.npz', figure,
             confirmation / 'summary.json', confirmation / 'manifest.json', confirmation / 'test_link_ranks.npz',
             lock_path, Path(lock['classifier_checkpoint']), state, acceptance]
    manifest = {
        'schema': 'verified_two_lp_delivery_v1',
        'tasks': {'sports-copurchase': {
            'training_kind': 'hard_four', 'validation_run': str(validation), 'confirmation_run': str(confirmation),
            'main_task_number': '07', 'independent_audit': str(audit_path), 'paper_mrr_pct': 37.68,
            'artifact_sha256': {str(p.resolve()): fingerprint(p) for p in bound}}},
        'seed0_test_mrr_pct': audit['test_mrr_pct'], 'seed0_gap_pp': audit['gap_pp'],
        'formal_repeat_mean_not_yet_measured': True, 'formal_repeat_score_not_guaranteed': True,
        'engineering_negative_policy_not_asserted_author_setting': True,
        'preserve_five_complete_repeat_results': True, 'new_downstream_seeds': list(range(1, 10)),
        'seed0_reused_not_retrained': True, 'entry': 'scripts/run_07_sports_repeats.py',
    }
    validate_delivery(manifest)
    output = ROOT / 'artifacts/remaining_main_pilots/two_lp_priority/verified_job07_sports_delivery.json'
    if output.exists() and read_json_file(output) != manifest:
        raise ValueError('Immutable numbered delivery already exists; refuse to replace it')
    atomic_write_json(output, manifest)
    print(json.dumps({'manifest': str(output), 'new_formal_GPU_jobs': 0,
                      'verified_seed0_test_pct': audit['test_mrr_pct']}))


if __name__ == '__main__':
    main()
