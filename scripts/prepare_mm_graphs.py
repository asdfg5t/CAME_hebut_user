"""Build immutable-source, memory-mapped MM-Graph training views for joint pretraining."""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from came_repro.author_loader import fingerprint
from came_repro.experiment import write_json

DATASETS = ['ele-fashion', 'sports-copurchase', 'cloth-copurchase', 'books-lp', 'books-nc']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=['cloth-copurchase', 'books-lp', 'books-nc'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw_root = ROOT / 'data/raw/mm_graph'
    manifest = json.loads((raw_root / 'manifest.json').read_text())
    text = json.loads((ROOT / 'artifacts/audit/clip_text_verification.json').read_text())
    images_path = ROOT / 'artifacts/audit/clip_image_verification.json'
    images = json.loads(images_path.read_text()) if images_path.exists() else {'datasets': {}}
    for dataset in args.datasets:
        directory = ROOT / 'data/processed/mm_graph' / dataset
        directory.mkdir(parents=True, exist_ok=True)
        completed_path = directory / 'manifest.json'
        if completed_path.exists():
            print(f'Prepared view exists: {dataset}; source/view hashes will be checked by loader', flush=True)
            continue
        names = ['clip_feat.pt'] + (['nc_edges-nodeid.pt', 'split.pt', 'labels-w-missing.pt'] if dataset in ('ele-fashion', 'books-nc') else ['lp-edge-split.pt'])
        source_hashes = {}
        for name in names:
            actual = fingerprint(raw_root / dataset / name)
            if actual != manifest['files'][dataset + '/' + name]['sha256']:
                raise ValueError(f'Raw source changed: {dataset}/{name}')
            source_hashes[name] = actual
        proof = text['datasets'][dataset]
        if not proof['text_block_verified'] or proof['text_block_index'] != 0:
            raise ValueError('Text-first feature layout has not been independently verified')
        features = torch.load(str(raw_root / dataset / 'clip_feat.pt'), map_location='cpu', weights_only=True, mmap=True)
        if features.shape[1] != 1024:
            raise ValueError('Unexpected feature width')
        nodes = len(features)
        del features
        gc.collect()
        if dataset in ('ele-fashion', 'books-nc'):
            edge = np.asarray(torch.load(raw_root / dataset / 'nc_edges-nodeid.pt', weights_only=True), dtype=np.int64)
            np.save(directory / 'labels.npy', np.asarray(torch.load(raw_root / dataset / 'labels-w-missing.pt', weights_only=True), dtype=np.int64))
            splits = torch.load(raw_root / dataset / 'split.pt', weights_only=True)
            for name, index in splits.items():
                np.save(directory / (name.replace('_idx', '') + '_indices.npy'), np.asarray(index, dtype=np.int64))
        else:
            split = torch.load(raw_root / dataset / 'lp-edge-split.pt', map_location='cpu', weights_only=False)
            edge = np.stack([np.asarray(split['train']['source_node'], dtype=np.int64), np.asarray(split['train']['target_node'], dtype=np.int64)], axis=1)
            del split
        np.save(directory / 'edge_index.npy', edge)
        result = {'dataset': dataset, 'nodes': nodes, 'edges': len(edge), 'revision': manifest['revision'],
                  'raw_sha256': source_hashes, 'layout': 'text_first_verified', 'features': [nodes, 1024],
                  'text_verification': proof, 'vision_verification': images['datasets'].get(dataset),
                  'feature_storage': {'kind': 'native_torch_mmap', 'source_relative_path': str((raw_root/dataset/'clip_feat.pt').relative_to(ROOT)),
                                      'source_sha256': source_hashes['clip_feat.pt'], 'precision': 'fp32', 'no_duplicate_feature_file': True},
                  'graph_policy': 'Official directed topology; for LP, ONLY training positive edges',
                  'storage_policy': 'Native torch mmap of preserved official features; no duplicate tensor file; no LP negative arrays retained in training RAM',
                  'files': {path.name: fingerprint(path) for path in directory.glob('*.npy')}}
        write_json(completed_path, result)
        del edge
        gc.collect()
        print(f'Prepared {dataset}: nodes={nodes}, train_topology_edges={result["edges"]}', flush=True)


if __name__ == '__main__':
    main()
