"""CPU-only independent source-number audits; never runs model training/test."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def audit_validation(run, task):
    import numpy as np
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from came_repro.author_loader import fingerprint
    from came_repro.prepared_data import prepared_directory
    from came_repro.filesystem_paths import windows_io_path
    run = Path(run); summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if summary['test_evaluations'] != 0 or any((run/name).exists() for name in ('test_link_ranks.npz', 'test_predictions.npz')):
        raise ValueError('Only validation-only candidates may be audited here')
    rows = [json.loads(x) for x in (run/'logs/metrics.jsonl').read_text().splitlines()]
    with (run/'logs/metrics.csv').open(newline='') as file: csv_rows = list(csv.DictReader(file))
    if len(rows) != len(csv_rows) or len(rows) != summary['epochs_executed']: raise ValueError('Epoch log count differs')
    for index, (row, other) in enumerate(zip(rows, csv_rows), 1):
        if row['step'] != index or not all(np.isfinite(v) and float(other[k]) == float(v) for k, v in row.items()):
            raise ValueError('CSV/JSONL full history differs')
    key = 'validation_mrr' if task == 'cloth' else 'val_accuracy'
    if rows[summary['selected_epoch']-1][key] != max(row[key] for row in rows): raise ValueError('Selected epoch differs')
    events = EventAccumulator(str(windows_io_path(run/'tensorboard')), size_guidance={'scalars': 0}); events.Reload()
    for tag in ('train_loss', key):
        values = events.Scalars(tag)
        if len(values) != len(rows) or not all(e.step == r['step'] and np.isclose(e.value, r[tag], rtol=1e-6, atol=1e-8) for e, r in zip(values, rows)):
            raise ValueError('TensorBoard differs')
    directory = Path(json.loads((run/'checkpoints/location.json').read_text())['checkpoint_directory'])
    checkpoint = directory/('link_head_last.pt' if task == 'cloth' else 'probe_last.pt')
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    if (state['history_rows'] != rows or state['selected_epoch'] != summary['selected_epoch']
            or state['next_epoch'] != len(rows)+1): raise ValueError('Full checkpoint history/cursor differs')
    if not all(k in state for k in ('optimizer', 'rng', 'sampling_generator', 'best_state')): raise ValueError('Incomplete resume state')
    if task == 'cloth':
        raw = ROOT/'data/raw/mm_graph/cloth-copurchase/lp-edge-split.pt'
        split = torch.load(str(raw), map_location='cpu', weights_only=False, mmap=True)['valid']
        with np.load(run/'validation_link_ranks.npz') as saved:
            for name in ('source_node', 'target_node'): np.testing.assert_array_equal(saved[name], split[name])
            rank = saved['rank']; assert len(rank) == 121323 and ((rank >= 1) & (rank <= 151)).all()
            value = float(np.mean(1./rank.astype(np.float64)))
        if abs(value-summary['validation_mrr']) > 1e-12: raise ValueError('Full official MRR recomputation differs')
    else:
        triples = np.load(prepared_directory('WN18RR')/'valid_triples.npy', mmap_mode='r')
        with np.load(run/'validation_predictions.npz') as saved:
            np.testing.assert_array_equal(saved['pairs'], triples[:, [0, 2]])
            np.testing.assert_array_equal(saved['labels'], triples[:, 1])
            value = float(np.mean(saved['prediction'] == saved['labels']))
        if len(triples) != 3034 or value != summary['validation_accuracy']: raise ValueError('Full official accuracy differs')
    figure = run/'figures'/('hard_lp_curves.png' if task == 'cloth' else 'probe_curves.png')
    return dict(passed=True, task=task, run=str(run), validation_score=value, epochs=len(rows), selected_epoch=summary['selected_epoch'],
        complete_official_validation_recomputed=True, CSV_JSONL_TensorBoard_full_checkpoint_verified=True,
        checkpoint_sha256=fingerprint(checkpoint), figure=str(figure), figure_sha256=fingerprint(figure),
        figure_visual_verified=False, test_evaluations=0)


def books_protocol_audit():
    import numpy as np
    import torch
    from came_repro.author_loader import fingerprint
    raw = ROOT/'data/raw/mm_graph/books-lp/lp-edge-split.pt'
    split = torch.load(str(raw), map_location='cpu', weights_only=False, mmap=True)
    n = 636502
    source, target = (np.asarray(split['train'][k]) for k in ('source_node', 'target_node'))
    degree = np.bincount(target, minlength=n)
    valid = split['valid']; ids = np.random.default_rng(218).choice(len(valid['source_node']), min(8192, len(valid['source_node'])), replace=False)
    positive = np.asarray(valid['target_node'])[ids]; negative = np.asarray(valid['target_node_neg'])[ids]
    if negative.shape[1] != 150 or negative.min() < 0 or negative.max() >= n: raise ValueError('Official negatives changed')
    previous = ROOT/'artifacts/audit/books_official_evaluator_rechecked_20261015.json'
    assert json.loads(previous.read_text())['passed']
    return dict(passed=True, stage='official_protocol_review_not_accuracy_repair', raw_sha256=fingerprint(raw),
        counts={k: dict(queries=len(v['source_node']), negatives=int(v['target_node_neg'].shape[1]) if 'target_node_neg' in v else None) for k, v in split.items()},
        train_validation_only_sample_queries=len(ids), positive_target_mean_train_in_degree=float(degree[positive].mean()),
        fixed_negative_target_mean_train_in_degree=float(degree[negative].mean()),
        authoritative_publisher='https://huggingface.co/datasets/mm-graph-org/mm-graph/tree/main/books-lp',
        official_evaluator_previously_verified_sha256=fingerprint(previous), test_scores_not_recomputed=True,
        author_negative_generation_and_downstream_head_still_unknown=True, paper_equivalence_resolved=False,
        artificial_score_reduction=False, new_GPU_jobs=0, new_test_evaluations=0)
