"""Read-only final protocol/identity audit; never starts or changes a GPU task."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.atomic_io import read_json_file, atomic_write_json
    verify_originals()
    sports = read_json_file(ROOT / 'artifacts/workflow_state/verified_job07_sports_delivery_repeats_current.json')
    queued = read_json_file(ROOT / 'artifacts/workflow_state/post_sports_main_repairs_current.json')
    for group in (sports['identity'], queued['identity']['sources']):
        for name, digest in group.items():
            if fingerprint(name) != digest:
                raise ValueError('Frozen active/queued file changed: ' + name)
    plan_path = ROOT / 'configs/post_sports_main_repairs_20261015.json'
    if fingerprint(plan_path) != queued['identity']['plan_sha256']:
        raise ValueError('Queued scientific plan changed')
    plan = read_json_file(plan_path)
    cloth = read_json_file(ROOT / 'configs/cloth_predictor_structure_20261015.json')
    reference = ROOT / cloth['reference_validation_run'] / 'summary.json'
    if fingerprint(reference) != cloth['reference_validation_summary_sha256']:
        raise ValueError('Cloth fixed validation reference changed')
    module_path = ROOT / 'scripts/run_cloth_predictor_candidate.py'
    spec = importlib.util.spec_from_file_location('cloth_final_audit', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recipe = module.structure_recipe(read_json_file(reference))
    assert recipe == dict(cloth['matched_recipe'], head_variant=cloth['candidate_head_variant'])
    assert recipe['max_epochs'] == 300 and recipe['patience'] == 10
    assert cloth['negative_pool_size'] == 4 and cloth['seed'] == 0
    assert cloth['fixed_negative_candidates'] == 150
    assert cloth['official_test_queries'] == 97456
    assert cloth['minimum_validation_gain_pp'] == .5
    assert cloth['engineering_adaptation_not_CAME_author_confirmed']
    wn = plan['WN']
    cp = Path(read_json_file(ROOT / wn['checkpoint_location_file'])['checkpoint_directory']) / 'probe_last.pt'
    assert fingerprint(cp) == wn['parent_checkpoint_sha256']
    assert wn['max_steps'] == 500 and wn['minimum_validation_gain_pp'] == .5
    assert wn['legacy_priority_pause_preserved']
    assert plan['Products']['full_nodes'] == 2449029
    assert plan['Products']['full_test_nodes'] == 2213091
    assert plan['Products']['width'] == 1024
    assert plan['Books']['artificial_degradation_or_metric_substitution_allowed'] is False
    assert plan['no_test_or_paper_target_used_for_model_selection']
    assert plan['no_CPU_neural_fallback_or_reduced_width_or_reduced_official_splits']
    report = {
        'passed': True, 'scope': 'final frozen identities and scientific protocol only, not new performance acceptance',
        'author_originals_verified': True, 'Sports_all_frozen_files_unchanged': True,
        'queued_all_frozen_files_unchanged': True,
        'Cloth_only_predictor_structure_changes': True,
        'WN_original_checkpoint_unchanged': True,
        'Products_full_dimensions_and_official_test_preserved': True,
        'validation_only_selection_and_locked_test_once': True,
        'Books_comparability_unresolved': True,
        'no_new_GPU_task_or_process_interruption': True,
        'no_active_or_queued_source_or_profile_changes': True,
        'new_test_results_generated': 0,
        'new_performance_guarantee': False,
        'plan_sha256': fingerprint(plan_path), 'audit_script_sha256': fingerprint(__file__),
        'Sports_statistics_at_read': sports.get('statistics', {}),
        'queued_state_at_read': queued['state'],
    }
    atomic_write_json(ROOT / 'artifacts/audit/final_main_repair_plan_verified_20261015.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__': main()
