"""Exact full-neighbor evaluation without retaining full-graph layer activations in RAM."""
import json
import shutil
import time
from pathlib import Path
from .pause_control import pause_if_requested
from .atomic_io import read_json_file

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from .author_loader import fingerprint
from .experiment import write_json


@torch.no_grad()
def batch_full_neighbor_embeddings(model, data, blocks, input_nodes):
    text, vision = data.modality_batch(input_nodes)
    early_mean=getattr(model,'representation_variant','came')=='without_modality_gnn'
    if early_mean:
        text=.5*(text+vision)
    h1=text.cuda()
    h2=h1 if early_mean else vision.cuda()
    for index, block in enumerate(blocks):
        block = block.to('cuda')
        h1 = F.gelu(model.gat_t5vit.norms[index](model.gat_t5vit.gat_layers[index](block, h1).flatten(1)))
        h2 = h1 if early_mean else F.gelu(model.gat_clip.norms[index](model.gat_clip.gat_layers[index](block, h2).flatten(1)))
    gate = model.fuse_gate(torch.cat([h1, h2], dim=-1))
    mixed = gate * h1 + (1 - gate) * h2
    return model.fuse_mlp(torch.cat([mixed, h1, h2], dim=-1))


def extract_full_embeddings(model, data, directory, checkpoint_sha256, batch_size=64, log=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / 'inference_state.json'
    target = directory / 'fused_embeddings.npy'
    width = model.gat_t5vit.norms[-1].normalized_shape[0]
    identity = {'checkpoint_sha256': checkpoint_sha256, 'data': data.audit, 'hidden_dim': width,
                'method': 'All inbound neighbors, all encoder layers, eval mode; no PPR approximation', 'precision': 'fp32'}
    state = read_json_file(state_path) if state_path.exists() else {'identity': identity, 'cursor': 0, 'complete': False}
    if state['identity'] != identity:
        raise ValueError('Inference cache belongs to another model/data/protocol')
    required = data.graph.num_nodes() * width * 4
    if not target.exists():
        from .storage_budget import guard_allocation
        guard_allocation(target, required + 1024**2)
    if not target.exists() and required + 3 * 1024**3 > shutil.disk_usage(directory).free:
        raise RuntimeError('Insufficient disk space for complete embeddings and reserve')
    if state['complete']:
        if fingerprint(target) != state['output_sha256']:
            raise ValueError('Complete inference cache checksum mismatch')
        return np.load(target, mmap_mode='r')
    if state['cursor'] and not target.exists():
        raise ValueError('Inference cursor exists but its saved embedding array is missing; refusing to skip nodes')
    output = np.lib.format.open_memmap(target, mode='r+' if target.exists() else 'w+', dtype=np.float32,
                                     shape=(data.graph.num_nodes(), width))
    graph = model._prep_graph(data.graph)
    if log:
        log.info('Exact inference resume cursor=%d/%d; checkpoint/data identity verified', state['cursor'], graph.num_nodes())
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = dgl.dataloading.DataLoader(graph, torch.arange(state['cursor'], graph.num_nodes()),
             dgl.dataloading.MultiLayerFullNeighborSampler(model.gat_t5vit.num_layers), batch_size=batch_size,
             shuffle=False, drop_last=False, num_workers=0)
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, (inputs, targets, blocks) in enumerate(loader):
            fused = batch_full_neighbor_embeddings(model, data, blocks, inputs)
            if not torch.isfinite(fused).all():
                raise FloatingPointError('Nonfinite inference output')
            output[targets.numpy()] = fused.cpu().numpy()
            output.flush()
            state['cursor'] = int(targets[-1]) + 1
            write_json(state_path, state)
            pause_if_requested()
            if log and ((batch_index+1) % 100 == 0 or state['cursor'] == graph.num_nodes()):
                log.info('Exact inference nodes=%d/%d elapsed=%.1fs', state['cursor'], graph.num_nodes(), time.perf_counter()-started)
    state['complete'] = state['cursor'] == graph.num_nodes()
    state['output_sha256'] = fingerprint(target)
    write_json(state_path, state)
    return np.load(target, mmap_mode='r')
