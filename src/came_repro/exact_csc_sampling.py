"""Exact all-inbound-neighbor blocks from immutable CSC, without a global DGL graph.

This is downstream exact message passing, not the pretraining PPR sampler.
Parallel edges and the certified one-loop-per-node topology are retained.
"""
import json
from pathlib import Path
import numpy as np
import torch


class ExactCSCGraph:
    def __init__(self, directory, verify=True):
        directory=Path(directory)
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if not manifest['complete']:raise ValueError('Complete certified CSC required')
        if verify:
            from .author_loader import fingerprint
            for name,digest in manifest['files'].items():
                if fingerprint(directory/name)!=digest:raise ValueError('Certified CSC file changed')
        self.nodes=manifest['identity']['nodes']
        self.indptr=np.load(directory/'indptr.npy',mmap_mode='r')
        self.indices=np.load(directory/'indices.npy',mmap_mode='r')
        if (len(self.indptr)!=self.nodes+1 or self.indptr[0]!=0
                or int(self.indptr[-1])!=len(self.indices) or len(self.indices)!=manifest['edges']):
            raise ValueError('Invalid complete CSC layout')
        self.idtype=torch.int64;self.device=torch.device('cpu')

    def num_nodes(self):return self.nodes
    def num_edges(self):return len(self.indices)

    def block(self, targets):
        import dgl
        targets=np.asarray(targets,dtype=np.int64)
        if targets.ndim!=1 or len(np.unique(targets))!=len(targets):raise ValueError('Unique one-dimensional targets required')
        if len(targets) and (targets.min()<0 or targets.max()>=self.nodes):raise ValueError('Target node out of range')
        lengths=self.indptr[targets+1].astype(np.int64)-self.indptr[targets].astype(np.int64)
        if (lengths<0).any():raise ValueError('Nonmonotonic CSC pointers')
        sources=[self.indices[int(self.indptr[t]):int(self.indptr[t+1])] for t in targets]
        incoming=np.concatenate(sources).astype(np.int64,copy=False) if len(targets) else np.empty(0,dtype=np.int64)
        if len(incoming) and (incoming.min()<0 or incoming.max()>=self.nodes):raise ValueError('CSC source ID out of range')
        # Destination order is a required prefix; other source nodes are sorted.
        extra=np.setdiff1d(np.unique(incoming),targets,assume_unique=True)
        source_nodes=np.concatenate((targets,extra))
        order=np.argsort(source_nodes)
        local_source=order[np.searchsorted(source_nodes[order],incoming)]
        local_target=np.repeat(np.arange(len(targets),dtype=np.int64),lengths)
        graph=dgl.create_block((torch.from_numpy(local_source),torch.from_numpy(local_target)),
                              num_src_nodes=len(source_nodes),num_dst_nodes=len(targets),idtype=torch.int64)
        graph.srcdata[dgl.NID]=torch.from_numpy(source_nodes)
        graph.dstdata[dgl.NID]=torch.from_numpy(targets.copy())
        return source_nodes,graph


class ExactCSCSampler:
    def __init__(self, layers):
        if layers!=2:raise ValueError('This acceptance covers the original two encoder layers')

    def sample_blocks(self,graph,targets):
        if not isinstance(graph,ExactCSCGraph):raise TypeError('Certified exact CSC graph required')
        if not isinstance(targets,torch.Tensor) or targets.dtype!=torch.int64 or targets.device.type!='cpu':
            raise ValueError('CPU int64 target IDs required')
        outputs=targets.clone();frontier=targets.numpy();blocks=[]
        for _ in range(2):
            frontier,block=graph.block(frontier);blocks.insert(0,block)
        return torch.from_numpy(frontier),outputs,blocks
