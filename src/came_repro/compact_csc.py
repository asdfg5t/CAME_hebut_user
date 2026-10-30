"""Exact incoming CSC: preserve parallel nonloops, replace all loops by one/node.

Only structural IDs use int32; neural features and weights remain full FP32.
"""
from pathlib import Path
import numpy as np


def build_index(edges, nodes, directory, source_sha256, chunk_size=1048576):
    from .atomic_io import atomic_write_json, atomic_replace
    from .author_loader import fingerprint
    import gc
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if edges.ndim != 2 or edges.shape[1] != 2 or not 0 < nodes < 2**31:
        raise ValueError('An Nx2 integral edge list and int32-safe node count required')
    if not np.issubdtype(edges.dtype, np.integer): raise ValueError('Integral IDs required')
    manifest_path = directory/'manifest.json'
    identity = dict(source_sha256=source_sha256, nodes=nodes,
                    method='stable_inbound_CSC_keep_parallel_nonloops_then_one_last_selfloop_v1', idtype='int32')
    if manifest_path.exists():
        import json
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        if saved['identity'] != identity: raise ValueError('CSC namespace identity changed')
        for name, sha in saved['files'].items():
            if fingerprint(directory/name) != sha: raise ValueError('Completed CSC changed')
        return saved
    degree = np.zeros(nodes, dtype=np.int64); removed = 0
    for start in range(0, len(edges), chunk_size):
        part = edges[start:start+chunk_size]
        if len(part) and (part.min() < 0 or part.max() >= nodes): raise ValueError('Node ID out of range')
        keep = part[:, 0] != part[:, 1]; removed += int((~keep).sum())
        degree += np.bincount(part[keep, 1], minlength=nodes)
    count = int(degree.sum())+nodes
    if count >= 2**31: raise ValueError('int32 edge-pointer overflow; no silent truncation')
    ptr = np.empty(nodes+1, dtype=np.int32); ptr[0] = 0
    ptr[1:] = np.cumsum(degree+1, dtype=np.int64)
    # Sorting IDs, rather than scipy sparse conversion, retains parallel edges.
    order = np.argsort(edges[:, 1], kind='stable')
    target = directory/'indices.build.npy'
    values = np.lib.format.open_memmap(target, mode='w+', dtype=np.int32, shape=(count,))
    processed = 0
    for start in range(0, len(order), chunk_size):
        pairs = edges[order[start:start+chunk_size]]
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        positions = processed+np.arange(len(pairs), dtype=np.int64)+pairs[:, 1]
        values[positions] = pairs[:, 0]
        processed += len(pairs)
    if processed != count-nodes: raise ValueError('CSC nonloop count mismatch')
    values[ptr[1:].astype(np.int64)-1] = np.arange(nodes, dtype=np.int32)
    values.flush(); values._mmap.close()
    ptr_path = directory/'indptr.build.npy'; np.save(ptr_path, ptr)
    del order, degree; gc.collect()
    atomic_replace(target, directory/'indices.npy'); atomic_replace(ptr_path, directory/'indptr.npy')
    report = dict(identity=identity, raw_edges=len(edges), edges=count, removed_selfloops=removed,
        files={n: fingerprint(directory/n) for n in ('indptr.npy', 'indices.npy')}, complete=True)
    atomic_write_json(manifest_path, report)
    return report


def load_graph(directory):
    import json
    import torch
    import dgl
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if not manifest['complete']: raise ValueError('Incomplete CSC index')
    # Copy-on-write mappings protect files while providing writable tensor views.
    ptr = np.load(directory/'indptr.npy', mmap_mode='c')
    values = np.load(directory/'indices.npy', mmap_mode='c')
    graph = dgl.graph(('csc', (torch.from_numpy(ptr), torch.from_numpy(values),
                              torch.empty(0, dtype=torch.int32))),
                      num_nodes=manifest['identity']['nodes'], idtype=torch.int32)
    return graph.formats('csc')
