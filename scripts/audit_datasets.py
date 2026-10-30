"""Verify download completeness and official dataset structure without training labels."""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import numpy as np
from came_repro.author_loader import fingerprint, verify_originals
from came_repro.data import load_dataset
from came_repro.experiment import write_json

EXPECTED_NODES = {'ele-fashion': 97766, 'sports-copurchase': 50250,
                  'cloth-copurchase': 125839, 'books-lp': 636502, 'books-nc': 685294}
PAPER_TABLE5_EDGES = {'ele-fashion': 199602, 'sports-copurchase': 356202,
                     'cloth-copurchase': 951271, 'books-lp': 3437017, 'books-nc': 7235084}


def lp_overlaps(splits, num_nodes):
    pairs = {key: (np.asarray(value['source_node'], dtype=np.int64), np.asarray(value['target_node'], dtype=np.int64))
             for key, value in splits.items()}
    ordered = {key: np.unique(source * num_nodes + target) for key, (source, target) in pairs.items()}
    undirected = {key: np.unique(np.minimum(source, target) * num_nodes + np.maximum(source, target))
                  for key, (source, target) in pairs.items()}
    return {key: {'ordered_train_overlap': int(len(np.intersect1d(ordered['train'], ordered[key], assume_unique=True))),
                  'undirected_train_overlap': int(len(np.intersect1d(undirected['train'], undirected[key], assume_unique=True)))}
            for key in ('valid', 'test')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=list(EXPECTED_NODES), default=list(EXPECTED_NODES))
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((ROOT / 'data/raw/mm_graph/manifest.json').read_text())
    incomplete = [key for key in manifest['selection'] if key not in manifest['files']]
    if incomplete:
        raise RuntimeError(f'Download incomplete: {incomplete}')
    for key, item in manifest['files'].items():
        path = ROOT / 'data/raw/mm_graph' / key
        if path.stat().st_size != item['size'] or fingerprint(path) != item['sha256']:
            raise RuntimeError(f'Data mismatch: {key}')
        if item.get('publisher_sha256') and item['publisher_sha256'] != item['sha256']:
            raise RuntimeError(f'Publisher hash mismatch: {key}')
    report = {'revision': manifest['revision'], 'verified_files': len(manifest['files']),
              'download_bytes': sum(item['size'] for item in manifest['files'].values()),
              'original_files': verify_originals(), 'datasets': {}}
    overlaps = {}
    for name in args.datasets:
        data = load_dataset(name, 'text_first_assumed')
        if data.graph.num_nodes() != EXPECTED_NODES[name]:
            raise RuntimeError(f'Paper node count mismatch: {name}')
        audit = data.audit
        audit['paper_table5_edges'] = PAPER_TABLE5_EDGES[name]
        audit['edge_count_delta_from_paper'] = audit['stored_training_graph_edges'] - PAPER_TABLE5_EDGES[name]
        if audit['edge_count_delta_from_paper']:
            print(f'WARNING: {name} official edge count differs from paper by {audit["edge_count_delta_from_paper"]}; original data retained.', flush=True)
        if data.labels is not None:
            audit['label_classes_present'] = torch.unique(data.labels).tolist()
            audit['negative_label_count'] = int((data.labels < 0).sum())
            audit['missing_labels_by_split'] = {key: int((data.labels[index] < 0).sum()) for key, index in data.splits.items()}
            audit['nodes_outside_official_splits'] = data.graph.num_nodes() - sum(len(value) for value in data.splits.values())
        else:
            audit['lp_split_fields'] = {key: sorted(value) for key, value in data.splits.items()}
            audit['total_positive_edges_across_splits'] = sum(len(value['source_node']) for value in data.splits.values())
            audit['lp_overlap'] = lp_overlaps(data.splits, data.graph.num_nodes())
            overlaps[name] = audit['lp_overlap']
        report['datasets'][name] = audit
        print(json.dumps(audit), flush=True)
        del data
        gc.collect()
    write_json(ROOT / 'artifacts/audit/dataset_audit.json', report)
    if overlaps:
        write_json(ROOT / 'artifacts/audit/lp_split_overlap.json', overlaps)
    print(f'PASS: {report["verified_files"]} publisher-verified files; {report["download_bytes"]/1e9:.2f} GB')


if __name__ == '__main__':
    main()
