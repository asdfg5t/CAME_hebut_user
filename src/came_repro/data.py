"""Validated official datasets and deterministic local forward-push PPR sampling."""
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import dgl
import numpy as np
import torch
from scipy.sparse import csr_matrix
from .author_loader import ROOT, fingerprint


@dataclass
class GraphData:
    graph: object
    features: torch.Tensor
    labels: object
    splits: dict
    audit: dict


def load_dataset(name, layout):
    directory = ROOT / 'data/raw/mm_graph' / name
    manifest = json.loads((directory.parent / 'manifest.json').read_text())
    names = ['clip_feat.pt'] + (['nc_edges-nodeid.pt', 'labels-w-missing.pt', 'split.pt']
                               if name in ('ele-fashion', 'books-nc') else ['lp-edge-split.pt'])
    fingerprints = {}
    for filename in names:
        key = name + '/' + filename
        entry = manifest['files'].get(key)
        if not entry or not entry['verified']:
            raise RuntimeError(f'Missing verified official file: {key}')
        actual = fingerprint(directory / filename)
        if actual != entry['sha256']:
            raise RuntimeError(f'Data checksum changed: {key}')
        fingerprints[filename] = actual
    # Publisher-verified .pt files are trusted. Feature tensors use the restricted loader.
    features = torch.load(directory / 'clip_feat.pt', map_location='cpu', weights_only=True)
    if features.ndim != 2 or features.shape[1] != 1024 or not torch.isfinite(features).all():
        raise ValueError('Expected finite 1024-dimensional two-part CLIP feature tensor')
    if layout not in ('text_first_assumed', 'image_first_assumed', 'text_first_verified', 'image_first_verified'):
        raise ValueError('Explicit feature layout required')
    n = len(features)
    labels = None
    if name in ('ele-fashion', 'books-nc'):
        edges = torch.as_tensor(torch.load(directory / 'nc_edges-nodeid.pt', weights_only=True), dtype=torch.long)
        labels = torch.as_tensor(torch.load(directory / 'labels-w-missing.pt', weights_only=True), dtype=torch.long).flatten()
        raw_split = torch.load(directory / 'split.pt', weights_only=True)
        splits = {key.replace('_idx', ''): torch.as_tensor(value, dtype=torch.long) for key, value in raw_split.items()}
        if len(labels) != n or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError('Label/edge shape mismatch')
        all_indices = torch.cat(list(splits.values()))
        if len(torch.unique(all_indices)) != len(all_indices):
            raise ValueError('Official splits overlap or contain duplicate IDs')
        if all_indices.min() < 0 or all_indices.max() >= n:
            raise ValueError('Split node ID out of range')
        source, target = edges.T
    else:
        # Large LP assets contain numpy arrays; only deserialize files checked above.
        splits = torch.load(directory / 'lp-edge-split.pt', map_location='cpu', weights_only=False)
        source = torch.as_tensor(splits['train']['source_node'], dtype=torch.long)
        target = torch.as_tensor(splits['train']['target_node'], dtype=torch.long)
    if source.min() < 0 or target.min() < 0 or max(source.max(), target.max()) >= n:
        raise ValueError('Graph node ID out of range')
    graph = dgl.graph((source, target), num_nodes=n)
    split_sizes = {key: (len(value) if labels is not None else len(value['source_node'])) for key, value in splits.items()}
    audit = {'dataset': name, 'revision': manifest['revision'], 'raw_sha256': fingerprints,
             'nodes': n, 'stored_training_graph_edges': graph.num_edges(), 'features': list(features.shape),
             'feature_layout': layout, 'layout_verified': layout.endswith('_verified'),
             'graph_policy': 'official directed NC edges; LP strictly train edges',
             'label_policy': 'labels not used in pretraining', 'splits': split_sizes}
    return GraphData(graph, features, labels, splits, audit)


class PPRSampler:
    """Approximate outgoing PPR with restart mass; top-k includes the seed.

    Alpha/epsilon are declared engineering assumptions. No BFS substitution is used.
    Residual mass is reported; exhausting max_pushes raises rather than silently truncating.
    """
    VERSION = 'forward_push_v1'

    def __init__(self, data, config):
        self.graph = data.graph
        self.config = config
        source, target = self.graph.edges()
        self.adjacency = csr_matrix((np.ones(len(source)), (source.numpy(), target.numpy())), shape=(self.graph.num_nodes(),) * 2)
        self.adjacency.sum_duplicates()
        if config['direction'] != 'official_directed':
            raise ValueError('Only explicit official directed topology currently implemented')
        identity = {'algorithm': self.VERSION, 'data': data.audit['raw_sha256'], 'config': config}
        import hashlib
        self.key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        self.directory = ROOT / 'data/processed/ppr' / self.key
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / 'manifest.json').write_text(json.dumps(identity, indent=2))

    def neighbors(self, seed):
        cache = self.directory / f'{seed}.npz'
        if cache.exists():
            with np.load(cache, allow_pickle=False) as stored:
                return stored['nodes'], float(stored['residual'])
        alpha, epsilon = self.config['alpha'], self.config['epsilon']
        if not 0 < alpha < 1 or epsilon <= 0:
            raise ValueError('Invalid PPR restart/tolerance')
        residual, scores = {seed: 1.0}, {}
        queue, queued = deque([seed]), {seed}
        pushes = 0
        while queue:
            node = queue.popleft()
            queued.remove(node)
            mass = residual.get(node, 0.0)
            start, end = self.adjacency.indptr[node:node + 2]
            weights = self.adjacency.data[start:end]
            degree = float(weights.sum())
            if mass <= epsilon * max(degree, 1):
                continue
            residual[node] = 0.0
            scores[node] = scores.get(node, 0.0) + alpha * mass
            destinations = self.adjacency.indices[start:end] if degree else np.array([seed])
            amounts = (1 - alpha) * mass * weights / degree if degree else np.array([(1 - alpha) * mass])
            for destination, amount in zip(destinations, amounts):
                destination = int(destination)
                residual[destination] = residual.get(destination, 0.0) + float(amount)
                a, b = self.adjacency.indptr[destination:destination + 2]
                threshold = epsilon * max(float(self.adjacency.data[a:b].sum()), 1)
                if residual[destination] > threshold and destination not in queued:
                    queue.append(destination)
                    queued.add(destination)
            pushes += 1
            if pushes >= self.config['max_pushes']:
                raise RuntimeError(f'PPR did not converge for seed={seed}; increase budget explicitly')
        ranking = sorted(scores, key=lambda node: (-scores[node], node))
        selected = [seed] + [node for node in ranking if node != seed][:self.config['top_k'] - 1]
        nodes = np.array(sorted(selected), dtype=np.int64)
        mass = sum(residual.values())
        np.savez_compressed(cache, nodes=nodes, residual=mass)
        return nodes, mass

    def sample(self, seeds):
        parts = [self.neighbors(int(seed)) for seed in seeds]
        nodes = np.unique(np.concatenate([part[0] for part in parts]))
        subgraph = dgl.node_subgraph(self.graph, torch.from_numpy(nodes))
        return subgraph, subgraph.ndata[dgl.NID], max(part[1] for part in parts)


def modality_features(features, layout):
    first, second = features[:, :512], features[:, 512:]
    return (first, second) if layout.startswith('text_first') else (second, first)
