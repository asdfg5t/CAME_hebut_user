"""Explicit full-coverage graph schedules and compact, persistent PPR caches."""
import hashlib
import json
import sqlite3
from collections import deque

import numpy as np
import torch
from scipy.sparse import csr_matrix
from .author_loader import ROOT
from .data import PPRSampler


class PackedPPRSampler(PPRSampler):
    VERSION = 'forward_push_int32_counts_int32_nodes_sqlite_v2'

    def __init__(self, data, config):
        self.graph = data.graph
        self.config = config
        if self.graph.num_nodes() > np.iinfo(np.int32).max:
            raise ValueError('Compact PPR storage requires node IDs below int32 range')
        source, target = self.graph.edges()
        # Integer multiplicities preserve the transition probabilities while saving RAM.
        self.adjacency = csr_matrix((np.ones(len(source), dtype=np.int32), (source.numpy(), target.numpy())), shape=(self.graph.num_nodes(),) * 2)
        self.adjacency.sum_duplicates()
        if config['direction'] != 'official_directed':
            raise ValueError('Direction must match the declared official prepared topology')
        topology = data.audit.get('view_files_sha256', {}).get('edge_index.npy') or data.audit.get('raw_sha256')
        ppr_config = {key: config[key] for key in ('top_k','alpha','epsilon','max_pushes','direction')}
        identity = {'algorithm': self.VERSION, 'topology_sha256': topology, 'nodes': self.graph.num_nodes(), 'config': ppr_config}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        directory = ROOT / 'data/processed/ppr_packed' / key
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'manifest.json').write_text(json.dumps(identity, indent=2))
        self.connection = sqlite3.connect(directory / 'neighbors.sqlite3')
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS ppr (seed INTEGER PRIMARY KEY, nodes BLOB NOT NULL, residual REAL NOT NULL)')
        self.pending = 0

    def neighbors(self, seed):
        row = self.connection.execute('SELECT nodes,residual FROM ppr WHERE seed=?', (seed,)).fetchone()
        if row:
            return np.frombuffer(row[0], dtype=np.int32).astype(np.int64), row[1]
        alpha, epsilon = self.config['alpha'], self.config['epsilon']
        if not 0 < alpha < 1 or epsilon <= 0:
            raise ValueError('Invalid PPR parameters')
        residual, scores = {seed: 1.0}, {}
        queue, queued = deque([seed]), {seed}
        pushes = 0
        while queue:
            node = queue.popleft()
            queued.remove(node)
            mass = residual.get(node, 0.)
            start, end = self.adjacency.indptr[node:node+2]
            weights = self.adjacency.data[start:end]
            degree = float(weights.sum())
            if mass <= epsilon * max(degree, 1):
                continue
            residual[node] = 0.
            scores[node] = scores.get(node, 0.) + alpha * mass
            destinations = self.adjacency.indices[start:end] if degree else np.array([seed])
            amounts = (1 - alpha) * mass * weights / degree if degree else np.array([(1-alpha)*mass])
            for destination, amount in zip(destinations, amounts):
                destination = int(destination)
                residual[destination] = residual.get(destination, 0.) + float(amount)
                a, b = self.adjacency.indptr[destination:destination+2]
                if residual[destination] > epsilon * max(float(self.adjacency.data[a:b].sum()), 1) and destination not in queued:
                    queue.append(destination)
                    queued.add(destination)
            pushes += 1
            if pushes >= self.config['max_pushes']:
                raise RuntimeError(f'PPR failed to converge for seed={seed}')
        ranked = sorted(scores, key=lambda node: (-scores[node], node))
        selected = [seed] + [node for node in ranked if node != seed][:self.config['top_k']-1]
        nodes = np.array(sorted(selected), dtype=np.int64)
        mass = sum(residual.values())
        self.connection.execute('INSERT INTO ppr VALUES(?,?,?)', (seed, nodes.astype(np.int32).tobytes(), mass))
        self.pending += 1
        if self.pending >= 256:
            self.flush()
        return nodes, mass

    def flush(self):
        self.connection.commit()
        self.pending = 0

    def close(self):
        self.flush()
        self.connection.close()


class JointSchedule:
    """Round-robin interleaving, exhausting each declared graph budget exactly once per epoch.

    The paper does not disclose mixing proportions or a seed budget. Both full coverage
    and declared capped coverage remain explicitly distinguished in configuration.
    """
    def __init__(self, node_counts, batch_size, generator, seed_limits=None):
        self.counts = dict(node_counts)
        self.batch_size = batch_size
        self.generator = generator
        self.seed_limits = seed_limits or {name: None for name in self.counts}
        self.epoch = 0
        self.orders = {}
        self.cursors = {}
        self.next_graph = 0
        self.new_epoch()

    def new_epoch(self):
        self.orders = {name: torch.randperm(count, generator=self.generator)[:self.seed_limits.get(name) or count] for name, count in self.counts.items()}
        self.cursors = {name: 0 for name in self.counts}
        self.next_graph = 0

    def next_batch(self):
        names = list(self.counts)
        for _ in names:
            name = names[self.next_graph]
            self.next_graph = (self.next_graph + 1) % len(names)
            start = self.cursors[name]
            if start < len(self.orders[name]):
                selected = self.orders[name][start:start+self.batch_size]
                self.cursors[name] += len(selected)
                return name, selected
        return None

    def state_dict(self):
        return {'epoch': self.epoch, 'orders': {name: value.clone() for name, value in self.orders.items()},
                'cursors': dict(self.cursors), 'next_graph': self.next_graph,
                'counts': dict(self.counts), 'batch_size': self.batch_size, 'seed_limits': dict(self.seed_limits),
                'generator': self.generator.get_state().clone()}

    def load_state_dict(self, state):
        for name in ('counts', 'batch_size', 'seed_limits'):
            if state[name] != getattr(self, name):
                raise ValueError(f'Joint schedule identity changed: {name}')
        self.epoch = state['epoch']
        self.orders = {name: value.clone() for name, value in state['orders'].items()}
        self.cursors = dict(state['cursors'])
        self.next_graph = state['next_graph']
        self.generator.set_state(state['generator'])
