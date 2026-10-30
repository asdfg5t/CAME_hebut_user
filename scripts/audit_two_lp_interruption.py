"""Read-only identity and full-checkpoint audit before restoring interrupted LP diagnostics."""
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import numpy as np
    import psutil
    import torch
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.quality_pipeline import best_resume
    from came_repro.storage_budget import check_budget

    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            command = ' '.join(process.info['cmdline'] or [])
            if process.info['pid'] != psutil.Process().pid and process.info['name'].lower() == 'python.exe' and any(
                    name in command for name in ('run_two_lp_hard_negative_validation.py', 'run_lp_hard_negative_candidate.py', 'prepare_two_lp_handoff.py')):
                raise RuntimeError('Existing diagnostic process found; refuse duplicate: ' + str(process.info['pid']))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if (ROOT / 'artifacts/workflow_state/two_lp_hard_negative_pause.json').exists():
        raise RuntimeError('Diagnostic pause exists; preserve it')
    verify_originals()
    probe = acquire_gpu_lock()
    probe.close()
    path = ROOT / 'artifacts/workflow_state/two_lp_hard_negative_current.json'
    ledger = read_json_file(path)
    identity = ledger['identity']
    for name, digest in identity['sources'].items():
        if fingerprint(ROOT / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    for name, key in [('scripts/run_two_lp_hard_negative_validation.py', 'self_sha256'),
                      ('scripts/run_lp_hard_negative_candidate.py', 'candidate_script_sha256'),
                      ('configs/lp_negative_objective_20261014.json', 'profile_sha256')]:
        if fingerprint(ROOT / name) != identity[key]:
            raise ValueError('Frozen candidate identity changed: ' + name)
    acceptance_path = ROOT / 'artifacts/audit/hard_lp_CUDA_acceptance_20261014.json'
    accepted = read_json_file(acceptance_path)
    if fingerprint(acceptance_path) != ledger['acceptance_sha256'] or not accepted['seed1_pause_resume_bitexact_weights_history']:
        raise ValueError('Original CUDA/checkpoint acceptance changed')
    completed_files = 0
    for rec in ledger['records'].values():
        if rec['state'] == 'completed':
            for name, digest in rec['artifact_sha256'].items():
                if fingerprint(name) != digest:
                    raise ValueError('Protected completed result changed: ' + name)
                completed_files += 1
    rec = ledger['records']['validate_cloth-copurchase']
    checkpoint = best_resume(rec, 'link_head_last.pt')
    if checkpoint is None:
        raise ValueError('No complete saved Cloth checkpoint; refuse fresh restart')
    run = Path(rec['attempts'][-1]['run_directory'])
    saved = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    if saved['schema'] != 'came_hard_lp_candidate_v1' or saved['seed'] != 0:
        raise ValueError('Wrong checkpoint schema or seed')
    if saved['config'] != read_json_file(run / 'config.json'):
        raise ValueError('Checkpoint and recorded config differ')
    required = ('optimizer', 'rng', 'sampling_generator', 'best_state', 'history_rows', 'data_audit')
    if any(name not in saved for name in required) or not saved['optimizer']['state']:
        raise ValueError('Incomplete optimizer/RNG/history state')
    complete = saved['completed_epochs']
    if saved['next_epoch'] != complete + 1 or len(saved['history_rows']) != complete:
        raise ValueError('Checkpoint epoch cursor/history disagreement')
    if [row['step'] for row in saved['history_rows']] != list(range(1, complete + 1)):
        raise ValueError('Checkpoint epoch history missing or duplicated')
    for key in ('head', 'best_state'):
        if not all(torch.isfinite(value).all() for value in saved[key].values()):
            raise ValueError('Nonfinite saved model state')
    for slot in saved['optimizer']['state'].values():
        if not all(torch.isfinite(value).all() for value in slot.values() if isinstance(value, torch.Tensor)):
            raise ValueError('Nonfinite saved optimizer state')
    if saved['config']['evaluation']['negative_pool_size'] != 4:
        raise ValueError('Frozen negative policy changed')
    np.random.default_rng().bit_generator.state = saved['sampling_generator']
    if not {'python', 'numpy', 'torch', 'cuda'}.issubset(saved['rng']):
        raise ValueError('RNG state incomplete')
    cache = Path(saved['config']['evaluation']['embedding_cache'])
    state = read_json_file(cache / 'inference_state.json')
    if not state['complete'] or state['identity']['data'] != saved['data_audit']:
        raise ValueError('Checkpoint data identity differs from complete cache')
    if fingerprint(cache / 'fused_embeddings.npy') != state['output_sha256']:
        raise ValueError('Frozen embedding output changed')
    raw = ROOT / 'data/raw/mm_graph/cloth-copurchase/lp-edge-split.pt'
    if fingerprint(raw) != saved['data_audit']['raw_sha256']['lp-edge-split.pt']:
        raise ValueError('Official split changed')
    encoder = Path(read_json_file(ROOT / 'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    if fingerprint(encoder) != state['identity']['checkpoint_sha256']:
        raise ValueError('Frozen encoder changed')
    budget = check_budget(additional_cache_bytes=256 * 2**20)
    report = {
        'checked_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'interruption_is_external_no_actual_diagnostic_process': True, 'GPU_lock_was_free': True,
        'user_explicitly_authorized_resume': True, 'previous_ledger_sha256': fingerprint(path),
        'frozen_source_profile_data_encoder_and_cache_verified': True, 'protected_completed_artifact_count': completed_files,
        'interrupted_run': str(run), 'checkpoint': str(checkpoint), 'parent_checkpoint_sha256': fingerprint(checkpoint),
        'completed_epochs': complete, 'next_epoch': complete + 1, 'selected_epoch': saved['selected_epoch'],
        'validation_mrr': saved['best_mrr'], 'complete_optimizer_RNG_generator_history': True,
        'budget': budget, 'resume_into_new_unique_run': True, 'new_formal_repeat_jobs': 0,
    }
    out = ROOT / 'artifacts/audit' / ('two_lp_external_interruption_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    atomic_write_json(out, report)
    print(json.dumps({'audit': str(out), 'completed_epochs': complete, 'next_epoch': complete + 1,
                      'parent_sha256': report['parent_checkpoint_sha256']}))


if __name__ == '__main__':
    main()
