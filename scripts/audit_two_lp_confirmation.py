"""Independently audit completed LP confirmations without running a GPU model."""
import argparse
import csv
import json
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def audit_confirmation(run, lock_path):
    import numpy as np
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.atomic_io import read_json_file
    from came_repro.filesystem_paths import windows_io_path

    run, lock_path = Path(run).resolve(), Path(lock_path).resolve()
    verify_originals()
    summary = read_json_file(run / 'summary.json')
    lock = read_json_file(lock_path)
    selected = Path(lock['validation_run_directory'])
    validation = read_json_file(selected / 'summary.json')
    dataset = summary['dataset']
    if dataset not in ('sports-copurchase', 'cloth-copurchase'):
        raise ValueError('Only the two priority tasks may be audited here')
    if summary['test_evaluations'] != 1 or not summary['confirmation_without_retraining']:
        raise ValueError('A single saved-head confirmation is required')
    for path, expected in [
        (lock_path, summary['selection_lock_sha256']),
        (selected / 'summary.json', lock['validation_summary_sha256']),
        (Path(lock['classifier_checkpoint']), lock['classifier_sha256']),
        (Path(lock['embedding_cache']) / 'inference_state.json', lock['inference_state_sha256']),
    ]:
        if fingerprint(path) != expected:
            raise ValueError('Locked artifact changed: ' + str(path))
    if not lock['locked_before_test'] or lock['test_used_for_selection']:
        raise ValueError('Validation-only model selection is required')
    if read_json_file(run / 'manifest.json')['parent_sha256'] != lock['classifier_sha256']:
        raise ValueError('Confirmation parent is not the selected saved head')
    raw = ROOT / 'data/raw/mm_graph' / dataset / 'lp-edge-split.pt'
    cache_state = read_json_file(Path(lock['embedding_cache']) / 'inference_state.json')
    if fingerprint(raw) != cache_state['identity']['data']['raw_sha256']['lp-edge-split.pt']:
        raise ValueError('Official data fingerprint changed')
    split = torch.load(str(raw), map_location='cpu', weights_only=False, mmap=True)
    n = cache_state['identity']['data']['nodes']
    train_keys = np.unique(np.asarray(split['train']['source_node'], dtype=np.int64) * n
                           + np.asarray(split['train']['target_node'], dtype=np.int64))
    metrics = {}
    for label, directory, key in [('validation', selected, 'valid'), ('test', run, 'test')]:
        with np.load(directory / (label + '_link_ranks.npz')) as saved:
            for name in ('source_node', 'target_node'):
                if not np.array_equal(saved[name], np.asarray(split[key][name])):
                    raise ValueError(label + ' official query order changed')
            reverse = np.isin(saved['target_node'].astype(np.int64) * n + saved['source_node'], train_keys)
            if not np.array_equal(reverse, saved['reverse_overlap']):
                raise ValueError(label + ' reverse overlap changed')
            ranks = saved['rank'].astype(np.float64)
            candidates = np.asarray(split[key]['target_node_neg']).shape[1]
            if candidates != 150 or not np.isfinite(ranks).all() or np.any((ranks < 1) | (ranks > 151)):
                raise ValueError('Official full candidate/rank bounds changed')
            metrics[label] = {
                'queries': len(ranks), 'negative_candidates': candidates,
                'mrr': float((1 / ranks).mean()),
                'hits_at_1': float((ranks <= 1).mean()),
                'hits_at_3': float((ranks <= 3).mean()),
                'hits_at_10': float((ranks <= 10).mean()),
                'reverse_overlap_queries': int(reverse.sum()),
                'nonoverlap_mrr': float((1 / ranks[~reverse]).mean()),
            }
    if abs(metrics['validation']['mrr'] - validation['validation_mrr']) > 2e-7:
        raise ValueError('Validation ranks do not match saved summary')
    if abs(metrics['test']['mrr'] - summary['test']['official']['mrr']) > 2e-7:
        raise ValueError('Test ranks do not match saved summary')
    rows = [json.loads(line) for line in (selected / 'logs/metrics.jsonl').read_text().splitlines()]
    with (selected / 'logs/metrics.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(rows) != len(csv_rows) or len(rows) != validation['epochs_executed']:
        raise ValueError('CSV/JSONL/full epoch counts disagree')
    for row, csv_row in zip(rows, csv_rows):
        for name, value in row.items():
            if not np.isfinite(float(value)) or float(csv_row[name]) != float(value):
                raise ValueError('CSV and JSONL numerical disagreement')
    if [int(row['step']) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Training history has duplicate or missing epochs')
    selected_row = rows[validation['selected_epoch'] - 1]
    if selected_row['validation_mrr'] != max(row['validation_mrr'] for row in rows):
        raise ValueError('Saved epoch was not chosen using maximum validation MRR')
    events = EventAccumulator(str(windows_io_path(selected / 'tensorboard')), size_guidance={'scalars': 0})
    events.Reload()
    for name in ('train_loss', 'validation_mrr'):
        values = events.Scalars(name)
        if len(values) != len(rows):
            raise ValueError('TensorBoard incomplete: ' + name)
        for event, row in zip(values, rows):
            if event.step != row['step'] or not np.isclose(event.value, row[name], rtol=1e-6, atol=1e-8):
                raise ValueError('TensorBoard disagrees with source metrics')
    figure_name = ('hard_lp_curves.png' if validation.get('negative_pool_size') == 4
                   else 'protocol_lp_curves.png' if validation.get('diagnostic_only')
                   else 'locked_lp_curves.png')
    figure = selected / 'figures' / figure_name
    if not figure.exists() or not (selected / 'logs/train.log').stat().st_size:
        raise ValueError('Training logger or figure missing')
    paper = 37.68 if dataset == 'sports-copurchase' else 28.88
    return {
        'dataset': dataset, 'confirmation_run': str(run), 'validation_run': str(selected),
        'validation_and_test_ranks_recomputed': metrics, 'official_raw_sha256': fingerprint(raw),
        'CSV_JSONL_TensorBoard_values_verified': True, 'full_epoch_count': len(rows),
        'selected_epoch': validation['selected_epoch'], 'saved_head_and_parent_SHA_verified': True,
        'figure': str(figure), 'figure_sha256': fingerprint(figure),
        'figure_visual_verified': False,
        'test_mrr_pct': metrics['test']['mrr'] * 100, 'paper_mrr_pct': paper,
        'gap_pp': metrics['test']['mrr'] * 100 - paper,
        'within_two_pp': abs(metrics['test']['mrr'] * 100 - paper) <= 2,
        'new_GPU_jobs': 0, 'formal_repeats_started': False,
    }


def main():
    from came_repro.atomic_io import atomic_write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--selection-lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_confirmation(args.run, args.selection_lock)
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
