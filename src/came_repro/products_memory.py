"""Read-only full CSC storage; only sampled miniblocks become int64 DGL graphs."""
import json
from pathlib import Path

import numpy as np


class MappedCsc:
    def __init__(self,directory):
        directory=Path(directory)
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if not manifest['complete']:raise ValueError('Complete frozen CSC required')
        self.nodes=int(manifest['identity']['nodes'])
        self.indptr=np.load(directory/'indptr.npy',mmap_mode='r',allow_pickle=False)
        self.indices=np.load(directory/'indices.npy',mmap_mode='r',allow_pickle=False)
        if (self.indptr.dtype!=np.int32 or self.indices.dtype!=np.int32
                or self.indptr.shape!=(self.nodes+1,) or self.indices.ndim!=1
                or self.indptr[0]!=0 or int(self.indptr[-1])!=len(self.indices)
                or not 0<self.nodes<2**31):raise ValueError('Verified compact CSC layout required')
        if len(self.indices)!=manifest['edges']:raise ValueError('CSC edge count changed')

    def num_nodes(self):return self.nodes

    @property
    def structural_bytes(self):return self.indptr.nbytes+self.indices.nbytes


class ReopenedNpyFeatures:
    """Copy only requested rows, then unmap immediately; OS cache stays reusable."""
    def __init__(self,path):
        self.path=Path(path)
        values=np.load(self.path,mmap_mode='r',allow_pickle=False)
        try:self.shape,self.dtype=values.shape,values.dtype
        finally:values._mmap.close()
        if len(self.shape)!=2 or self.shape[1]!=512 or self.dtype!=np.float32:
            raise ValueError('Verified complete FP32 Products text features required')

    def __len__(self):return self.shape[0]

    def __getitem__(self,index):
        values=np.load(self.path,mmap_mode='r',allow_pickle=False)
        try:return np.array(values[index],copy=True)
        finally:values._mmap.close()


def cache_node_ids(graph,capacity):
    """Label-free row priority: exact full-graph source occurrence count."""
    if capacity<1:raise ValueError('Positive exact feature-cache capacity required')
    nodes=graph.num_nodes();counts=np.zeros(nodes,dtype=np.int64)
    for start in range(0,len(graph.indices),2**20):
        counts+=np.bincount(np.asarray(graph.indices[start:start+2**20],dtype=np.int64),minlength=nodes)
    order=np.lexsort((np.arange(nodes,dtype=np.int64),-counts))
    return np.sort(order[:min(capacity,nodes)])


class ExactFeatureCache:
    """Exact FP32 GPU hits plus exact CPU-staged misses; neural work stays CUDA."""
    def __init__(self,source,ids,device='cuda'):
        import torch
        self.source=source;self.ids=np.asarray(ids,dtype=np.int64)
        if source.dtype!=np.float32 or source.shape[1]!=512:raise ValueError('Unchanged FP32 width512 inputs required')
        if len(np.unique(self.ids))!=len(self.ids) or (self.ids<0).any() or (self.ids>=len(source)).any():raise ValueError('Distinct official cache IDs required')
        self.lookup=np.full(len(source),-1,dtype=np.int32);self.lookup[self.ids]=np.arange(len(ids),dtype=np.int32)
        self.values=torch.empty((len(ids),512),dtype=torch.float32,device=device)
        for start in range(0,len(ids),8192):
            self.values[start:start+8192]=torch.from_numpy(np.array(source[self.ids[start:start+8192]],copy=True)).to(device)
        self.hits,self.queries=0,0

    def modality_batch(self,nodes):
        import torch
        index=nodes.cpu().numpy() if isinstance(nodes,torch.Tensor) else np.asarray(nodes,dtype=np.int64)
        cached=self.lookup[index];mask=cached>=0
        output=torch.empty((len(index),512),dtype=torch.float32,device=self.values.device)
        for present in [True,False]:
            positions=np.flatnonzero(mask if present else ~mask)
            if not len(positions):continue
            if present:
                rows=torch.as_tensor(cached[positions].astype(np.int64),device=self.values.device)
                values=self.values[rows]
            else:values=torch.from_numpy(np.array(self.source[index[positions]],copy=True)).to(self.values.device)
            output[torch.as_tensor(positions,device=self.values.device)]=values
        self.hits+=int(mask.sum());self.queries+=len(index)
        return output,output

    def stats(self):
        import hashlib
        return dict(cached_rows=len(self.ids),cached_bytes=self.values.numel()*self.values.element_size(),
            ids_sha256=hashlib.sha256(self.ids.tobytes()).hexdigest(),rows_requested=self.queries,
            cache_hits=self.hits,hit_fraction=self.hits/self.queries if self.queries else None,
            precision='fp32',feature_width=512,neural_device=str(self.values.device),selection_uses_labels=False)
def memory_snapshot():
    import psutil
    vm=psutil.virtual_memory();proc=psutil.Process().memory_info()
    return dict(total_RAM_bytes=vm.total,available_RAM_bytes=vm.available,
        process_RSS_bytes=proc.rss,process_private_bytes=getattr(proc,'private',proc.rss))


def require_cuda():
    """Same actual Torch/DGL check without importing unrelated LP pipelines."""
    import torch
    import dgl
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required; no CPU fallback')
    graph=dgl.graph(([0],[1]),num_nodes=2).to('cuda')
    graph.ndata['x']=torch.ones(2,1,device='cuda')
    graph.update_all(dgl.function.copy_u('x','m'),dgl.function.sum('m','y'))
    if graph.ndata['y'][1].item()!=1:raise RuntimeError('Actual DGL CUDA failed')
