"""Complete graph data backed by read-only disk features for multi-graph training."""
import json
from dataclasses import dataclass
from pathlib import Path

import dgl
import numpy as np
import torch
from .author_loader import ROOT, fingerprint


@dataclass
class PreparedData:
    graph: object
    features: object
    labels: object
    splits: dict
    audit: dict

    def modality_batch(self, nodes):
        index = nodes.cpu().numpy() if isinstance(nodes, torch.Tensor) else np.asarray(nodes)
        values = self.features[index].float() if isinstance(self.features, torch.Tensor) else torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True))
        if values.shape[1] == 512:
            return values, values
        return values[:, :512], values[:, 512:]


def prepared_directory(dataset):
    group = 'mm_graph' if dataset in ('ele-fashion', 'sports-copurchase', 'cloth-copurchase', 'books-lp', 'books-nc') else 'text_graphs'
    return ROOT / 'data/processed' / group / dataset


def load_prepared(dataset, require_verified=True, build_graph=True):
    directory = prepared_directory(dataset)
    manifest = json.loads((directory / 'manifest.json').read_text())
    multimodal = 'raw_sha256' in manifest
    if multimodal:
        text_path = ROOT / 'artifacts/audit/clip_text_verification.json'
        vision_path = ROOT / 'artifacts/audit/clip_image_verification.json'
        text = json.loads(text_path.read_text())['datasets'].get(dataset, {})
        vision = json.loads(vision_path.read_text())['datasets'].get(dataset, {}) if vision_path.exists() else {}
        if require_verified and (not text.get('text_block_verified') or not vision.get('vision_encoder_verified')):
            raise RuntimeError(f'Both modality encoders must be independently verified before joint training: {dataset}')
        if text.get('text_block_index') != 0:
            raise ValueError('Prepared multimodal views require verified text-first order')
        feature_name = 'clip_features.npy'
        layout = 'verified_text_first'
    else:
        state = json.loads((directory / 'clip_encoding_state.json').read_text())
        if not state['complete'] or state['cursor'] != manifest['nodes']:
            raise RuntimeError(f'Complete CUDA CLIP features are not ready: {dataset}')
        feature_name = 'clip_text.npy'
        layout = 'CLIP_text_reused_as_visual_placeholder_per_paper_5.2'
    for filename, expected in manifest['files'].items():
        if fingerprint(directory / filename) != expected:
            raise ValueError(f'Prepared view changed: {dataset}/{filename}')
    if multimodal and manifest.get('feature_storage', {}).get('kind') == 'native_torch_mmap':
        storage = manifest['feature_storage']
        source_path = ROOT / storage['source_relative_path']
        if fingerprint(source_path) != storage['source_sha256']:
            raise ValueError('Official raw feature source changed')
        features = torch.load(str(source_path), map_location='cpu', weights_only=True, mmap=True)
    else:
        features = np.load(directory / feature_name, mmap_mode='r', allow_pickle=False)
    if len(features) != manifest['nodes'] or features.shape[1] not in (512, 1024):
        raise ValueError('Feature shape mismatch')
    edges = np.load(directory / 'edge_index.npy', mmap_mode='r', allow_pickle=False)
    if build_graph:
        source = torch.tensor(np.asarray(edges[:, 0]), dtype=torch.long)
        target = torch.tensor(np.asarray(edges[:, 1]), dtype=torch.long)
        graph = dgl.graph((source, target), num_nodes=manifest['nodes'])
    else:
        graph = None
    labels_path = directory / 'labels.npy'
    labels = np.load(labels_path, mmap_mode='r') if labels_path.exists() else None
    splits = {path.name.removesuffix('_indices.npy'): np.load(path, mmap_mode='r') for path in directory.glob('*_indices.npy')}
    audit = {'dataset': dataset, 'nodes': manifest['nodes'], 'graph_edges': len(edges), 'features': list(features.shape),
             'layout': layout, 'prepared_manifest_sha256': fingerprint(directory / 'manifest.json'),
             'view_files_sha256': manifest['files'], 'raw_sha256': manifest.get('raw_sha256', manifest['raw_source']['files'] if 'raw_source' in manifest else {}),
             'feature_provenance': {'text': text, 'vision': vision} if multimodal else manifest['clip_encoder'],
             'graph_policy': manifest.get('graph_policy') or manifest.get('text_policy'),
             'storage': 'Read-only mmap; only selected node feature rows copied to CUDA'}
    return PreparedData(graph, features, labels, splits, audit)
