"""CPU preflight: freeze two finite candidates and preserve the interrupted Products cursor."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import numpy as np
    import psutil
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.storage_budget import check_budget
    verify_originals()
    matching = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(process.info['cmdline'] or [])
            if any(name in cmd for name in ('run_products_single_validation.py', 'run_post_sports_main_repairs.py', 'run_07_sports_repeats.py')):
                matching.append({'pid': process.pid, 'cmdline': cmd})
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    if matching: raise RuntimeError('Existing training/supervisor still active: ' + json.dumps(matching))
    lock = acquire_gpu_lock(); lock.close()
    product_path = ROOT / 'artifacts/workflow_state/products_single_seed_current.json'
    product = read_json_file(product_path)
    cache = Path(product['embedding_cache']); state_path = cache / 'inference_state.json'
    state = read_json_file(state_path)
    output = cache / 'fused_embeddings.npy'
    matrix = np.load(output, mmap_mode='r'); offset = matrix.offset
    assert matrix.shape == (2449029, 1024) and matrix.dtype == np.float32
    matrix._mmap.close()
    checksum = hashlib.sha256()
    with output.open('rb') as stream:
        stream.seek(offset); remaining = state['cursor'] * 1024 * 4
        while remaining:
            chunk = stream.read(min(8 * 2**20, remaining))
            if not chunk: raise ValueError('Truncated Products saved prefix')
            checksum.update(chunk); remaining -= len(chunk)
    if checksum.hexdigest() != state['prefix_sha256']: raise ValueError('Products durable prefix mismatch')
    old_supervisor = ROOT / 'artifacts/workflow_state/post_sports_main_repairs_current.json'
    for name, digest in read_json_file(old_supervisor)['identity']['sources'].items():
        if fingerprint(name) != digest: raise ValueError('Prior frozen source changed')
    cloth_run = ROOT / 'runs/20261015T044740Z_cloth_head_s_878f21_s0_8f49d4'
    wn_run = ROOT / 'runs/20261014T045920Z_kg_affine_WN_fce445_s0_cb5aeb'
    cloth = read_json_file(cloth_run / 'summary.json'); wn = read_json_file(wn_run / 'summary.json')
    best = Path(read_json_file(cloth_run / 'checkpoints/location.json')['checkpoint_directory']) / 'link_head_best.pt'
    crows = [json.loads(s) for s in (cloth_run / 'logs/metrics.jsonl').read_text().splitlines()]
    wrows = [json.loads(s) for s in (wn_run / 'logs/metrics.jsonl').read_text().splitlines()]
    assert cloth['selected_epoch'] == 97 and cloth['epochs_executed'] == 107
    assert len(crows) == 107 and len(wrows) == wn['epochs_executed']
    validation_rows = [float(r['val_accuracy']) for r in wrows[-100:]]
    evidence = {'Cloth_last_20_train_BCE_first_last': [crows[-20]['train_loss'], crows[-1]['train_loss']],
                'Cloth_constant_lr': sorted(set(r['lr_used'] for r in crows)),
                'WN_last_100_validation_accuracy_min_max': [min(validation_rows), max(validation_rows)],
                'source_is_train_validation_not_test': True}
    atomic_write_json(ROOT / 'artifacts/audit/priority_cloth_wn_train_validation_evidence_20261016.json', evidence)
    plan = {
        'objective': 'User-priority bounded Cloth and WN diagnostics before formal repeats',
        'encoder_sha256': cloth['encoder_sha256'], 'candidate_order': ['wn', 'cloth'],
        'new_candidates_per_task': 1, 'seed': 0, 'test_used_for_selection': False,
        'minimum_validation_gain_pp': .5, 'absolute_paper_gap_target_pp': 2, 'score_guaranteed': False,
        'cloth': {'reference_run': str(cloth_run.relative_to(ROOT)), 'reference_summary_sha256': fingerprint(cloth_run / 'summary.json'),
                  'reference_validation': cloth['validation_mrr'], 'initial_best_head_sha256': fingerprint(best),
                  'recipe': {'head_variant': cloth['head_variant'], 'lr': .0001, 'batch_size': 4096,
                             'max_epochs': 80, 'patience': 20, 'loss_reduction': 'separate_mean_sum'},
                  'change': 'New refinement phase, validation-selected best head initialization; new Adam at one-tenth LR. Explicit new optimizer, not identical resume.',
                  'full_nodes': 125839, 'full_validation_queries': 121323, 'full_test_queries': 97456,
                  'fixed_test_negatives': 150, 'negative_pool_size': 4},
        'wn': {'reference_run': str(wn_run.relative_to(ROOT)), 'reference_summary_sha256': fingerprint(wn_run / 'summary.json'),
               'reference_validation': wn['validation_accuracy'],
               'recipe': {'classes': 11, 'lr': .05, 'max_epochs': 2000, 'patience': 100, 'lr_patience': 20, 'min_lr': .00005},
               'change': 'Fresh fixed seed0 linear classifier; initial paper lr=.05, train CE, validation-loss ReduceLROnPlateau factor=.1. No nonlinear classifier or feature change.',
               'full_nodes': 40943, 'pair_width': 2048, 'full_test_queries': 3134},
        'evidence': 'artifacts/audit/priority_cloth_wn_train_validation_evidence_20261016.json',
        'paper_evidence': 'Appendix C: frozen encoder, head/tail concatenation linear KG classifier, CE, validation-selected early stopping; original LP head unpublished',
        'engineering_adaptations': ['Cloth warm-start refinement and lower LR', 'WN validation-loss LR scheduler', 'Patience choices are reconstructed'],
        'unknowns': ['Author optimizer/scheduler/LP head not confirmed', 'Frozen representation quality may cap transfer; optimizer changes may fail'],
        'resources': {'one_GPU_serial': True, 'additional_embedding_bytes': 0, 'checkpoint_metadata_reservation_bytes': 512 * 2**20, 'CPU_neural_fallback': False},
        'acceptance': ['Actual torch AND DGL CUDA', 'Both tasks seed1 optimizer/scheduler/RNG/history resume equivalence',
                       'Full official validation gain>=0.5pp; reject otherwise without new test', 'SHA-locked saved head full official test once',
                       'Independent complete predictions/ranks, CSV/JSONL/TensorBoard/checkpoint/parent hashes, visual PNG review'],
        'protection': ['All six completed formal repeated tasks', 'Author originals', 'Raw data', 'Existing models/runs', 'Products prefix cursor'],
        'Products_action': 'Inactive after external interruption; preserve verified prefix and prevent old auto-resume until priority pair is complete/audited',
        'formal_repeats_started_by_assistant': 0, 'periodic_automation': False,
    }
    path = ROOT / 'configs/priority_cloth_wn_20261016.json'
    if path.exists() and read_json_file(path) != plan: raise ValueError('Existing priority plan differs; audit before replacing')
    atomic_write_json(path, plan)
    pause = ROOT / 'artifacts/workflow_state/post_sports_main_repairs_pause.json'
    if not pause.exists(): atomic_write_json(pause, {'reason': 'user_priority_Cloth_WN_20261016', 'Products_prefix_preserved': True, 'do_not_resume_old_queue_while_priority_pair_active': True})
    report = {'passed': True, 'no_existing_matching_process': True, 'global_GPU_lock_free': True,
              'Products_cursor': state['cursor'], 'Products_prefix_SHA_verified': checksum.hexdigest(),
              'Products_inference_state_sha256': fingerprint(state_path), 'Products_old_ledger_sha256': fingerprint(product_path),
              'old_supervisor_ledger_sha256': fingerprint(old_supervisor),
              'existing_frozen_sources_and_originals_verified': True, 'plan_sha256': fingerprint(path),
              'budget': check_budget(additional_cache_bytes=512 * 2**20), 'GPU_tasks_started': 0,
              'external_exit_cause': 'Unknown; no Python process remains, running ledger alone is stale',
              'original_ledger_or_cache_modified': False}
    atomic_write_json(ROOT / 'artifacts/audit/priority_cloth_wn_preflight_20261016.json', report)
    print(json.dumps({'passed': True, 'Products_cursor': state['cursor'], 'validation_evidence': evidence}))


if __name__ == '__main__': main()
