"""Verify complete prepared assets, immutable author files, CUDA and storage before training."""
import datetime
import gc
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from came_repro.author_loader import verify_originals
from came_repro.experiment import write_json
from came_repro.joint_train import validate_joint
from came_repro.prepared_data import load_prepared
from came_repro.storage_budget import check_budget


def main():
    expected = ROOT / '.venv-came39/Scripts/python.exe'
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError('Use the designated .venv-came39 interpreter')
    report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'original_files': verify_originals(), 'executable': sys.executable,
              'storage': check_budget(additional_cache_bytes=2 * 2**30), 'datasets': {}}
    for name in ('joint_representation_reconstruction', 'joint_few_shot_reconstruction'):
        config = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
        validate_joint(config)
        if config['training']['max_steps'] is not None or any(value is not None for value in config['sampling']['seed_limits'].values()):
            raise ValueError('Default training configurations must retain complete declared coverage')
    datasets = ('ele-fashion', 'sports-copurchase', 'cloth-copurchase', 'books-lp', 'books-nc',
                'ogbn-products', 'ogbn-arxiv', 'wiki-cs', 'FB15K237', 'WN18RR')
    for name in datasets:
        print('Verifying all prepared file hashes: ' + name, flush=True)
        data = load_prepared(name, require_verified=True, build_graph=False)
        indices = np.random.default_rng(42).choice(data.audit['nodes'], min(128, data.audit['nodes']), replace=False)
        text, image = data.modality_batch(indices)
        for values in (text.numpy(), image.numpy()):
            if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1., atol=2e-5):
                raise ValueError('Nonfinite or non-normalized CLIP feature sample: ' + name)
        report['datasets'][name] = data.audit
        del data, text, image
        gc.collect()
    subprocess.run([sys.executable, str(ROOT / 'scripts/check_environment.py')], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    report['cuda'] = json.loads((ROOT / 'artifacts/audit/environment.json').read_text())
    report['ready_for_user_training'] = True
    report['paper_reproduction_complete'] = False
    write_json(ROOT / 'artifacts/audit/preflight_reproduction.json', report)
    print('PREPARATION_READY: complete assets, author integrity, actual torch/DGL CUDA and budgets passed.', flush=True)


if __name__ == '__main__':
    main()
