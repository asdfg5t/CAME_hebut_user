"""CPU-only verification of the two completed validation-rejected candidates."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal-reviewed-figures', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.filesystem_paths import windows_io_path
    from came_repro.kg_graph_probe import kg_input
    verify_originals()
    ledger = read_json_file(ROOT / 'artifacts/workflow_state/priority_cloth_wn_current.json')
    reports = {}
    for task in ('wn', 'cloth'):
        if task not in ledger['rejections']:
            raise ValueError('Only completed rejected candidates may be audited here')
        record = ledger['records']['validate_' + task]
        run = Path(record['run_directory'])
        summary = read_json_file(run / 'summary.json')
        if summary['test_evaluations'] != 0 or any((run / n).exists() for n in ('test_predictions.npz', 'test_link_ranks.npz')):
            raise ValueError('Rejected candidate unexpectedly has test output')
        for name, digest in record['artifact_sha256'].items():
            if fingerprint(name) != digest:
                raise ValueError('Completed artifact changed: ' + name)
        key = 'val_accuracy' if task == 'wn' else 'validation_mrr'
        with (run / 'logs/metrics.csv').open(newline='', encoding='utf-8') as stream:
            rows_csv = list(csv.DictReader(stream))
        rows = [json.loads(line) for line in (run / 'logs/metrics.jsonl').read_text().splitlines()]
        assert len(rows) == len(rows_csv) == summary['epochs_executed']
        for index, (row, row_csv) in enumerate(zip(rows, rows_csv), 1):
            assert row['step'] == index
            assert all(np.isfinite(v) and float(row_csv[k]) == float(v) for k, v in row.items())
        assert rows[summary['selected_epoch'] - 1][key] == max(r[key] for r in rows)
        events = EventAccumulator(str(windows_io_path(run / 'tensorboard')), size_guidance={'scalars': 0})
        events.Reload()
        for key_event in ('train_loss', key, 'lr_used', 'lr_next'):
            values = events.Scalars(key_event)
            assert len(values) == len(rows)
            assert all(e.step == r['step'] and np.isclose(e.value, r[key_event], rtol=1e-6, atol=1e-8) for e, r in zip(values, rows))
        checkpoint = Path(read_json_file(run / 'checkpoints/location.json')['checkpoint_directory']) / 'priority_head_last.pt'
        state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
        assert state['history_rows'] == rows and state['next_epoch'] == len(rows) + 1
        assert state['selected_epoch'] == summary['selected_epoch']
        assert all(k in state for k in ('optimizer', 'scheduler', 'rng', 'sampling_generator', 'best_state', 'data_audit'))
        if task == 'cloth':
            split = torch.load(str(ROOT / 'data/raw/mm_graph/cloth-copurchase/lp-edge-split.pt'), map_location='cpu', weights_only=False, mmap=True)['valid']
            with np.load(run / 'validation_link_ranks.npz') as saved:
                np.testing.assert_array_equal(saved['source_node'], split['source_node'])
                np.testing.assert_array_equal(saved['target_node'], split['target_node'])
                assert len(saved['rank']) == 121323
                value = float(np.mean(1. / saved['rank'].astype(np.float64)))
            assert abs(value - summary['validation_mrr']) < 1e-12
        else:
            cache = Path(summary['embedding_cache'])
            official = kg_input('WN18RR', np.load(cache / 'fused_embeddings.npy', mmap_mode='r'), 'valid')
            with np.load(run / 'validation_predictions.npz') as saved:
                np.testing.assert_array_equal(saved['pairs'], official.pairs)
                np.testing.assert_array_equal(saved['labels'], official.labels)
                value = float(np.mean(saved['prediction'] == official.labels))
            assert len(official) == 3034 and value == summary['validation_accuracy']
        figure = run / ('figures/priority_probe_curves.png' if task == 'wn' else 'figures/hard_lp_curves.png')
        report = {'task': task, 'run': str(run), 'validation_only': True, 'test_evaluations': 0,
                  'validation_score': value, 'gain_pp': ledger['rejections'][task]['validation_gain_pp'],
                  'complete_official_validation_verified': True, 'epochs': len(rows),
                  'CSV_JSONL_TensorBoard_complete_checkpoint_RNG_verified': True,
                  'checkpoint_sha256': fingerprint(checkpoint), 'figure': str(figure),
                  'figure_sha256': fingerprint(figure), 'figure_visual_verified': False,
                  'passed_validation_gate': False, 'historical_test_results_preserved': True}
        path = ROOT / ('artifacts/audit/priority_' + task + '_rejected_independent_20261016.json')
        if args.seal_reviewed_figures:
            assert read_json_file(path)['figure_sha256'] == report['figure_sha256']
            report['figure_visual_verified'] = True
        atomic_write_json(path, report)
        reports[task] = report
    print(json.dumps(reports))


if __name__ == '__main__':
    main()
