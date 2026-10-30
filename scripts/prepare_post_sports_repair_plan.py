"""CPU-only identity/resource audit before queuing authorized diagnostics."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import torch
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.storage_budget import check_budget
    verify_originals()
    sports = read_json_file(ROOT / 'artifacts/workflow_state/verified_job07_sports_delivery_repeats_current.json')
    sources = sports['identity']
    for name, digest in sources.items():
        if fingerprint(name) != digest:
            raise ValueError('Active Sports frozen source changed: ' + name)
    old = ROOT / 'runs/20261014T075324Z_wn_wolfe_con_4e29af_s0_434d61'
    saved = Path(read_json_file(old / 'checkpoints/location.json')['checkpoint_directory']) / 'probe_last.pt'
    state = torch.load(str(saved), map_location='cpu', weights_only=False)
    assert state['schema'] == 'came_wolfe_linear_v1'
    assert state['config'] == read_json_file(old / 'config.json')
    assert state['completed_epochs'] + 1 == state['next_epoch']
    assert len(state['history_rows']) == state['completed_epochs']
    assert state['config']['evaluation']['max_epochs'] == 500
    for key in ('optimizer', 'rng', 'classifier', 'best_state'):
        assert key in state
    def finite(value):
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            assert torch.isfinite(value).all(), 'Nonfinite checkpoint tensor'
        elif isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, (list, tuple)):
            for child in value: finite(child)
    finite(state)
    accepted = read_json_file(ROOT / 'artifacts/audit/convergence_cuda_acceptance_20261014.json')
    assert accepted['passed'] and accepted['solver_pause_resume_bitexact_weights_and_history']
    assert accepted['script_sha256'] == fingerprint(ROOT / 'scripts/run_convergence_candidate.py')
    assert accepted['profile_sha256'] == fingerprint(ROOT / 'configs/convergence_repair_20261014.json')
    reference = 'runs/20261014T045920Z_kg_affine_WN_fce445_s0_cb5aeb'
    product = read_json_file(ROOT / 'configs/products_single_seed_20261014.json')
    budget = check_budget(additional_cache_bytes=product['resources']['new_embedding_bytes'] + 512 * 2**20)
    files = [
        'scripts/run_cloth_predictor_candidate.py', 'configs/cloth_predictor_structure_20261015.json',
        'scripts/run_lp_hard_negative_candidate.py', 'configs/lp_negative_objective_20261014.json',
        'scripts/audit_two_lp_confirmation.py', 'scripts/run_convergence_candidate.py',
        'configs/convergence_repair_20261014.json', 'scripts/run_products_single_validation.py',
        'configs/products_single_seed_20261014.json', 'configs/storage.json',
        'artifacts/audit/convergence_cuda_acceptance_20261014.json',
        'artifacts/workflow_state/products_capacity_current.json',
    ]
    plan = {
        'objective': 'One bounded Cloth candidate, resume unfinished WN candidate, full Products single seed; audit Books protocol separately',
        'start_after': 'User Sports final ten-seed ledger and free global GPU lock',
        'frozen_infrastructure': files, 'head_metadata_reservation_bytes': 512 * 2**20,
        'WN': {'reference_validation_run': reference,
               'reference_validation_summary_sha256': fingerprint(ROOT / reference / 'summary.json'),
               'checkpoint_location_file': str((old / 'checkpoints/location.json').relative_to(ROOT)),
               'parent_checkpoint_sha256': fingerprint(saved), 'minimum_validation_gain_pp': .5,
               'max_steps': 500, 'legacy_priority_pause_preserved': True},
        'Cloth': {'source_review': 'artifacts/audit/cloth_predictor_structure_review_20261015.json',
                  'finite_candidates': 1, 'minimum_validation_gain_pp': .5, 'max_epochs': 300, 'patience': 10},
        'Products': {'full_nodes': 2449029, 'full_test_nodes': 2213091, 'width': 1024,
                     'new_embedding_bytes': product['resources']['new_embedding_bytes'],
                     'epochs': 2000, 'patience': 100, 'actual_capacity_calibration_required': True},
        'Books': {'review': 'artifacts/audit/books_lp_candidate_integrity_review_20261014.json',
                  'author_negative_generation_and_head_unknown': True,
                  'artificial_degradation_or_metric_substitution_allowed': False, 'new_GPU_runs': 0},
        'selection': 'Full official validation only; fixed seed0; >=0.5pp validation improvement then SHA-locked saved head test once',
        'paper_parameters_separate_from_engineering': True,
        'unresolved': ['CAME author LP head/optimizer unpublished', 'Exact Books evaluation protocol not confirmed',
                       'Products full-inference slowest batch capacity/throughput unknown', 'Future test proximity not guaranteed'],
        'no_test_or_paper_target_used_for_model_selection': True,
        'no_CPU_neural_fallback_or_reduced_width_or_reduced_official_splits': True,
        'formal_repeats_started_by_assistant': 0, 'periodic_automation': False,
        'protected': ['Author originals', 'Raw datasets', 'All prior results/checkpoints', 'Active Sports frozen source/config/seeds'],
    }
    path = ROOT / 'configs/post_sports_main_repairs_20261015.json'
    if path.exists() and read_json_file(path) != plan:
        raise ValueError('Existing plan changed; inspect before replacing')
    atomic_write_json(path, plan)
    audit = {'CPU_only_no_GPU_job_started': True, 'originals_verified': True,
             'Sports_frozen_sources_all_verified': True, 'WN_checkpoint_sha256': fingerprint(saved),
             'WN_completed_steps': state['completed_epochs'], 'WN_next_step': state['next_epoch'],
             'WN_optimizer_RNG_history_config_finite_verified': True,
             'plan_sha256': fingerprint(path), 'budget': budget,
             'Sports_current_statistics': sports.get('statistics', {})}
    atomic_write_json(ROOT / 'artifacts/audit/post_sports_main_repairs_preflight_20261015.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k != 'budget'}, ensure_ascii=False))


if __name__ == '__main__': main()
