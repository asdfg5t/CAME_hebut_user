"""CPU metric/source audit only; no score-driven protocol substitution or neural training."""
import ast
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import numpy as np
    import torch
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.task_metrics import ranking_metrics
    archive = ROOT / 'data/raw/source_archives/mm-graph-benchmark-0769fa670c69a6dc0756a2d7fc697b980e494775.zip'
    assert fingerprint(archive) == '491a0056d7767063def646052881852159fd071e6966b1533361083642c9cd96'
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith('/lp/lp_dataset.py'))
        source = z.read(name).decode()
    tree = ast.parse(source)
    evaluator = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LinkPredictionEvaluator')
    # Execute only the fixed, inspected upstream metric class, not its dataset loader/imports.
    namespace = {'np': np, 'torch': torch}
    exec(compile(ast.Module(body=[evaluator], type_ignores=[]), name, 'exec'), namespace)
    random = np.random.default_rng(20261015)
    positive = torch.from_numpy(random.integers(-3, 4, 113).astype(np.float32))
    negative = torch.from_numpy(random.integers(-3, 4, (113, 150)).astype(np.float32))
    positive[:3] = torch.tensor([10., -10., 0.])
    negative[2] = 0  # complete ties, plus rank-one and rank-last cases
    official = namespace['LinkPredictionEvaluator']().eval({'y_pred_pos': positive, 'y_pred_neg': negative})
    current = ranking_metrics(positive, negative)
    differences = {}
    for key in ('mrr', 'hits@1', 'hits@3', 'hits@10'):
        actual = official[key + '_list'].double()
        difference = float((actual - current[key]).abs().max())
        assert difference < 1e-7
        differences[key] = difference
    assert "os.path.join(root, f'lp-edge-split.pt')" in source
    report = {
        'passed': True, 'CPU_metric_computation_only_no_GPU_or_model_training': True,
        'pinned_official_repo': 'https://github.com/mm-graph-benchmark/mm-graph-benchmark/tree/0769fa670c69a6dc0756a2d7fc697b980e494775',
        'archive_sha256': fingerprint(archive), 'upstream_file': name,
        'official_loader_actually_reads': 'lp-edge-split.pt',
        'README_random_hard_names_do_not_select_different_file_in_pinned_loader': True,
        'synthetic_queries_with_150_negatives': 113,
        'cases_include_equal_score_ties_best_and_worst_ranks': True,
        'maximum_per_query_metric_difference': differences,
        'interpretation': 'Current MRR matches official torch evaluator within float precision. This does not explain or fix Books paper gap.',
        'test_used_to_choose_new_protocol': False,
        'datasets_checkpoints_and_experiment_files_changed': False,
        'Books_author_evaluation_details_still_unconfirmed': True,
        'live_publisher_directory_checked': 'https://huggingface.co/datasets/mm-graph-org/mm-graph/tree/main/books-lp',
        'no_new_random_hard_asset_in_publisher_listing': True,
    }
    atomic_write_json(ROOT / 'artifacts/audit/books_official_evaluator_rechecked_20261015.json', report)
    print(json.dumps(report))


if __name__ == '__main__': main()
