"""Read-only, full selected-pool PPR integrity and context-coverage verification."""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import numpy as np
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint
    ledger = read_json_file(ROOT / 'artifacts/workflow_state/representation_coverage_current.json')
    attempt = ledger['records']['fresh_pretraining']['attempts'][-1]
    if not attempt.get('run_directory'): raise ValueError('Actual production run not initialized')
    run = Path(attempt['run_directory'])
    plan = read_json_file(ROOT / 'configs/representation_coverage_validation_20261016.json')
    config = read_json_file(run / 'config.json')
    previous = read_json_file(ROOT / 'artifacts/audit/pretraining_context_coverage_review_20261014.json')
    results = {}
    for reference in previous['datasets']:
        name = reference['dataset']; pool = run / ('sampling/' + name + '_seed_pool.npy')
        completed = run / ('sampling/' + name + '_offline_ppr.json')
        if not completed.exists(): raise ValueError('All offline PPR pools must finish before this audit')
        identity = plan['pools'][name]
        if fingerprint(pool) != identity['sha256']: raise ValueError('Production unlabeled pool differs from frozen plan')
        seeds = np.load(pool); manifest = Path(plan['clones'][name]['private']).parent / 'manifest.json'
        cache_identity = read_json_file(manifest)
        if cache_identity['config'] != {k: config['sampling'][k] for k in ('top_k','alpha','epsilon','max_pushes','direction')}:
            raise ValueError('PPR settings differ from declared candidate')
        if cache_identity['topology_sha256'] != read_json_file(reference['cache_manifest'])['topology_sha256']:
            raise ValueError('PPR graph changed')
        context = np.zeros(reference['full_nodes'], dtype=bool); counts = []
        with sqlite3.connect(Path(plan['clones'][name]['private']).as_uri() + '?mode=ro', uri=True) as database:
            for seed in seeds:
                row = database.execute('SELECT nodes FROM ppr WHERE seed=?', (int(seed),)).fetchone()
                if not row: raise ValueError('Missing offline PPR row')
                nodes = np.frombuffer(row[0], dtype=np.int32)
                if (not 1 <= len(nodes) <= 128 or not np.all(nodes[1:] > nodes[:-1])
                    or np.any(nodes < 0) or np.any(nodes >= len(context)) or seed not in nodes):
                    raise ValueError('Invalid PPR node IDs')
                context[nodes] = True; counts.append(len(nodes))
        unique = int(context.sum())
        if unique < reference['unique_cached_PPR_context_nodes']:
            raise ValueError('Superset pool unexpectedly lost original contextual coverage')
        results[name] = {'full_nodes': len(context), 'unlabeled_seeds': len(seeds), 'all_rows_verified': True,
            'context_nodes': unique, 'context_coverage_pct': 100*unique/len(context),
            'old_context_nodes': reference['unique_cached_PPR_context_nodes'],
            'mean_neighborhood_nodes': float(np.mean(counts)), 'full_graph_topology_unchanged': True}
    path = ROOT / 'artifacts/audit/representation_coverage_offline_PPR_verified_20261016.json'
    atomic_write_json(path, {'production_run': str(run), 'complete_selected_pool_verified': True, 'results': results,
        'test_labels_used': False, 'context_coverage_not_effective_learning_or_score_guarantee': True})
    print(json.dumps(results))


if __name__ == '__main__': main()
