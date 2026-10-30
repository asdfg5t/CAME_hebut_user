"""Independent CPU-only audit of completed priority diagnostics; never trains or retests."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal-reviewed-figures', action='store_true', help='Only after the calling assistant has actually viewed the exact saved figure hashes')
    parser.add_argument('--after-pid', type=int, help='One-shot dependency: audit after this existing diagnostic controller exits; never restart it')
    args = parser.parse_args()
    if args.after_pid:
        import psutil
        try:
            process = psutil.Process(args.after_pid)
            if 'run_priority_cloth_wn_validation.py' not in ' '.join(process.cmdline()):
                raise ValueError('PID is not the authorized diagnostic controller')
            print('WAITING_FOR_EXISTING_DIAGNOSTIC_EXIT; CPU audit only; GPU jobs=0.', flush=True)
            process.wait()
        except psutil.NoSuchProcess:
            pass
        state = json.loads((ROOT / 'artifacts/workflow_state/priority_cloth_wn_current.json').read_text())['state']
        if state != 'completed_bounded_priority_diagnostics_pending_manual_delivery_review':
            print('DEFERRED: diagnostic did not complete; no retraining or retesting.', flush=True)
            return
    import numpy as np
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.filesystem_paths import windows_io_path
    from came_repro.kg_graph_probe import kg_input
    from audit_two_lp_confirmation import audit_confirmation
    verify_originals()
    ledger_path = ROOT / 'artifacts/workflow_state/priority_cloth_wn_current.json'
    ledger = read_json_file(ledger_path)
    for name, digest in ledger['identity']['sources'].items():
        if fingerprint(name) != digest: raise ValueError('Frozen diagnostic source changed')
    directory = ROOT / 'artifacts/remaining_main_pilots/priority_cloth_wn_20261016'
    audits = {}
    for task in ('wn', 'cloth'):
        confirmed = ledger['records'].get('confirm_' + task, {})
        if confirmed.get('state') != 'completed': continue
        selected = ledger['records']['validate_' + task]
        run = Path(selected['run_directory']); final = Path(confirmed['run_directory'])
        lock_path = directory / (task + '_selection_lock.json'); lock = read_json_file(lock_path)
        for name, expected in ((run / 'summary.json', lock['validation_summary_sha256']),
                               (Path(lock['classifier_checkpoint']), lock['classifier_sha256']),
                               (Path(lock['embedding_cache']) / 'inference_state.json', lock['inference_state_sha256'])):
            if fingerprint(name) != expected: raise ValueError('Selected artifact changed')
        if not lock['locked_before_test'] or lock['test_used_for_selection'] is not False: raise ValueError('Invalid test selection protocol')
        if read_json_file(final / 'manifest.json')['parent_sha256'] != lock['classifier_sha256']: raise ValueError('Wrong confirmation parent')
        for record in (selected, confirmed):
            for name, digest in record['artifact_sha256'].items():
                if fingerprint(name) != digest: raise ValueError('Completed diagnostic artifact changed')
        if task == 'cloth':
            audit = audit_confirmation(final, lock_path)
        else:
            cache = Path(lock['embedding_cache']); matrix = np.load(cache / 'fused_embeddings.npy', mmap_mode='r')
            split_values = {}
            for name, saved_path, split in (('validation', run / 'validation_predictions.npz', 'valid'), ('test', final / 'test_predictions.npz', 'test')):
                official = kg_input('WN18RR', matrix, split)
                with np.load(saved_path) as saved:
                    if not np.array_equal(saved['pairs'], official.pairs) or not np.array_equal(saved['labels'], official.labels): raise ValueError('Complete official WN pair/label mismatch')
                    split_values[name] = {'queries': len(official), 'accuracy': float(np.mean(saved['prediction'] == official.labels))}
            if split_values['test']['queries'] != 3134 or split_values['test']['accuracy'] != confirmed['summary']['test_accuracy']: raise ValueError('Independent full-test accuracy mismatch')
            if split_values['validation']['accuracy'] != selected['summary']['validation_accuracy']: raise ValueError('Independent validation accuracy mismatch')
            rows = [json.loads(line) for line in (run / 'logs/metrics.jsonl').read_text().splitlines()]
            with (run / 'logs/metrics.csv').open(newline='', encoding='utf-8') as stream: csv_rows = list(csv.DictReader(stream))
            if len(rows) != len(csv_rows) or len(rows) != selected['summary']['epochs_executed']: raise ValueError('WN history count mismatch')
            for row, saved in zip(rows, csv_rows):
                for key, value in row.items():
                    if not np.isfinite(value) or float(saved[key]) != float(value): raise ValueError('WN CSV/JSONL numerical mismatch')
            if [r['step'] for r in rows] != list(range(1, len(rows) + 1)): raise ValueError('WN missing/duplicate epochs')
            selected_epoch = selected['summary']['selected_epoch']
            if rows[selected_epoch - 1]['val_accuracy'] != max(r['val_accuracy'] for r in rows): raise ValueError('WN selected epoch not validation best')
            events = EventAccumulator(str(windows_io_path(run / 'tensorboard')), size_guidance={'scalars': 0}); events.Reload()
            for key in ('train_loss', 'val_accuracy', 'val_loss', 'lr_next'):
                values = events.Scalars(key)
                if len(values) != len(rows): raise ValueError('WN missing TensorBoard steps')
                if any(e.step != r['step'] or not np.isclose(e.value, r[key], rtol=1e-6, atol=1e-8) for e, r in zip(values, rows)): raise ValueError('WN TensorBoard numbers disagree')
            checkpoint = Path(read_json_file(run / 'checkpoints/location.json')['checkpoint_directory']) / 'priority_head_last.pt'
            saved = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
            if saved['completed_epochs'] != len(rows) or saved['next_epoch'] != len(rows) + 1 or saved['history_rows'] != rows: raise ValueError('WN final complete checkpoint/history mismatch')
            for key in ('optimizer', 'scheduler', 'rng', 'sampling_generator', 'best_state'):
                if key not in saved: raise ValueError('WN checkpoint state missing')
            figure = run / 'figures/priority_probe_curves.png'
            value = split_values['test']['accuracy'] * 100
            audit = {'dataset': 'WN18RR', 'confirmation_run': str(final), 'validation_run': str(run),
                     'full_official_validation_test_pairs_labels_verified': True, 'independent_values': split_values,
                     'CSV_JSONL_TensorBoard_checkpoint_verified': True, 'parent_selected_weight_SHA_verified': True,
                     'test_accuracy_pct': value, 'paper_accuracy_pct': 81.97, 'gap_pp': value - 81.97,
                     'within_two_pp': abs(value - 81.97) <= 2, 'figure': str(figure), 'figure_sha256': fingerprint(figure)}
        audit['figure_visual_verified'] = False
        audits[task] = audit
    existing = directory / 'verified_handoff_audit.json'
    prior = read_json_file(existing) if existing.exists() else {}
    if args.seal_reviewed_figures:
        if not audits: raise ValueError('No completed confirmation figures to review')
        for task, audit in audits.items():
            previous = prior.get('tasks', {}).get(task, {})
            if previous.get('figure_sha256') != audit['figure_sha256']: raise ValueError('First write the pending audit, view its exact figures, then seal')
            audit['figure_visual_verified'] = True
    eligible = [task for task, audit in audits.items() if audit['within_two_pp'] and audit['figure_visual_verified']]
    report = {'ledger_state_at_read': ledger['state'], 'source_ledger_sha256': fingerprint(ledger_path),
              'tasks': audits, 'ready_for_training_entry_preparation': eligible,
              'not_confirmed_or_rejected': [t for t in ('wn', 'cloth') if t not in audits],
              'validation_rejections': ledger.get('rejections', {}),
              'full_formal_repeat_training_started': False, 'no_GPU_model_inference_training_or_retest': True}
    atomic_write_json(existing, report)
    print(json.dumps({'confirmed': list(audits), 'eligible_after_visual_review': eligible, 'ledger_state': ledger['state']}))


if __name__ == '__main__': main()
