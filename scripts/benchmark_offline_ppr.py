"""Bounded CPU PPR timing on independent unlabeled seeds; no model training."""
import gc
import json
import sys
import time
from pathlib import Path
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from came_repro.author_loader import ROOT, verify_originals
from came_repro.experiment import write_json
from came_repro.prepared_data import load_prepared
from came_repro.joint_sampling import PackedPPRSampler


def main():
    config = json.loads((ROOT / 'configs/subset_offline_24h.json').read_text())
    generator = np.random.default_rng(20261011)
    records = []
    for dataset in config['datasets']:
        data = load_prepared(dataset, require_verified=True)
        sampler = PackedPPRSampler(data, config['sampling'])
        rows = []
        try:
            for seed in generator.choice(data.graph.num_nodes(), size=64, replace=False).tolist():
                cached = bool(sampler.connection.execute('SELECT 1 FROM ppr WHERE seed=?', (seed,)).fetchone())
                started = time.perf_counter()
                nodes, residual = sampler.neighbors(seed)
                rows.append({'seed': seed, 'cached_before': cached, 'seconds': time.perf_counter() - started,
                             'neighbor_count': len(nodes), 'ppr_residual': residual})
            sampler.flush()
            uncached = [row['seconds'] for row in rows if not row['cached_before']]
            record = {'dataset': dataset, 'samples': rows, 'uncached_samples': len(uncached),
                      'uncached_mean_seconds': float(np.mean(uncached)) if uncached else None,
                      'uncached_p95_seconds': float(np.percentile(uncached, 95)) if uncached else None,
                      'full_pool_size': config['sampling']['seed_limits'][dataset],
                      'topology_sha256': data.audit['view_files_sha256']['edge_index.npy']}
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != 'samples'}), flush=True)
        finally:
            sampler.close()
        del sampler, data
        gc.collect()
    write_json(ROOT / 'artifacts/audit/offline_ppr_timing_20261011.json', {'sampling_seed': 20261011,
               'purpose': 'Independent uniform unlabeled timing sample; preparation only, not learning quality',
               'originals': verify_originals(), 'records': records,
               'full_pool_preparation_estimate_seconds': sum(row['full_pool_size'] * (row['uncached_mean_seconds'] or 0) for row in records),
               'estimate_is_not_a_completion_guarantee': True})


if __name__ == '__main__':
    main()
