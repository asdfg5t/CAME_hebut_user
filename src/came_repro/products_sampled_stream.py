"""Declared sampled Products inference with full official scoring and bounded storage."""
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from .atomic_io import atomic_write_json, atomic_replace, unique_temporary_path, retry_windows_io
from .author_loader import ROOT, fingerprint
from .experiment import Experiment, rng_state, restore_rng, seed_everything
from .pause_control import pause_if_requested, TrainingPaused
from .products_memory import MappedCsc,ReopenedNpyFeatures,ExactFeatureCache,cache_node_ids,memory_snapshot

OUT = ROOT / 'artifacts/products_sampled_stream'


class ProductsExperiment(Experiment):
    """Keep the full storage check; release its large directory walk in a worker."""
    def __init__(self,*args,**kwargs):
        from . import storage_budget
        from .products_controller_storage import external_budget_report
        previous=storage_budget.check_budget
        storage_budget.check_budget=external_budget_report
        try:super().__init__(*args,**kwargs)
        finally:storage_budget.check_budget=previous


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def estimate_seconds(rate, plan):
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('Measured positive throughput required')
    b = plan['budget']
    return plan['nodes'] / rate * b['inference_safety_factor'] + b['head_and_setup_reserve_seconds']


class RowCursor:
    """Append-only rows; fsync precedes the atomic cursor commit."""
    def __init__(self, checkpoint, identity, shape, dtype, resume=None, allocation_guard=None):
        self.path = Path(checkpoint)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.identity = dict(identity, shape=list(shape), dtype=self.dtype.str)
        self.digest = hashlib.sha256()
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError('Nonempty rows x columns required')
        if resume:
            state = read(resume)
            if state['identity'] != self.identity:
                raise ValueError('Row cursor source/model/protocol identity changed')
            self.values_path, self.cursor = Path(state['values_path']), state['cursor']
            if type(self.cursor) is not int or not 0 <= self.cursor <= shape[0]:
                raise ValueError('Row cursor out of range')
        else:
            self.cursor = 0
            self.values_path = self.path.parent / (self.path.stem + '_values.npy')
            if self.values_path.exists():
                raise ValueError('Refuse to overwrite old rows')
            if allocation_guard:
                allocation_guard(self.values_path, int(np.prod(shape))*self.dtype.itemsize+4096)
            a = np.lib.format.open_memmap(self.values_path, mode='w+', shape=self.shape, dtype=self.dtype)
            a.flush()
            a._mmap.close()
        a = np.load(self.values_path, mmap_mode='r')
        if a.shape != self.shape or a.dtype != self.dtype:
            raise ValueError('Row matrix shape or dtype changed')
        self.offset = a.offset
        a._mmap.close()
        self.row_bytes = self.shape[1]*self.dtype.itemsize
        with self.values_path.open('rb') as f:
            f.seek(self.offset)
            left = self.cursor*self.row_bytes
            while left:
                block = f.read(min(left, 8*2**20))
                if not block:
                    raise ValueError('Truncated committed row prefix')
                self.digest.update(block)
                left -= len(block)
        if resume and self.digest.hexdigest() != state['prefix_sha256']:
            raise ValueError('Committed row prefix changed')
        self.persist()

    def persist(self):
        atomic_write_json(self.path, dict(identity=self.identity, cursor=self.cursor,
            values_path=str(self.values_path), prefix_sha256=self.digest.hexdigest()))

    def commit(self, start, values):
        values = np.asarray(values, dtype=self.dtype)
        if (values.ndim != 2 or values.shape[1] != self.shape[1] or start != self.cursor
                or self.cursor+len(values) > self.shape[0] or not np.isfinite(values).all()):
            raise ValueError('Invalid/noncontiguous/nonfinite row commit')
        blob = values.tobytes()
        with self.values_path.open('r+b') as f:
            f.seek(self.offset+self.cursor*self.row_bytes)
            if f.write(blob) != len(blob):
                raise OSError('Incomplete row write')
            f.flush()
            os.fsync(f.fileno())
        self.digest.update(blob)
        self.cursor += len(values)
        self.persist()

    def complete_values(self):
        if self.cursor != self.shape[0]:
            raise ValueError('Cannot report an incomplete official matrix')
        return np.load(self.values_path, mmap_mode='r')


def reserve(report):
    from .storage_budget import size_bytes
    settings = read(ROOT/'configs/storage.json')
    root = Path(settings['large_checkpoint_root'])
    owned = lambda: sum(size_bytes(x) for x in root.iterdir() if x.is_dir() and '_products_fas_' in x.name)
    atomic_write_json(OUT/'storage_reservation.json', dict(root=str(root.resolve()),
        baseline_cache=report['external_cache_bytes'], baseline_owned=owned(),
        reserved=report['requested_cache_bytes'], cap=int(settings['external_cache_high_water_gib']*2**30),
        minimum_free=int(settings['reserve_free_gib']*2**30)))


def allocation_guard(path, additional):
    from .storage_budget import size_bytes
    r = read(OUT/'storage_reservation.json')
    root, path = Path(r['root']).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or len(path.relative_to(root).parts) != 2 or '_products_fas_' not in path.parent.name:
        raise ValueError('Products write escaped its bounded checkpoint family')
    owned = sum(size_bytes(x) for x in root.iterdir() if x.is_dir() and '_products_fas_' in x.name)
    if (additional < 0 or max(0, owned-r['baseline_owned'])+additional > r['reserved']
            or r['baseline_cache']-r['baseline_owned']+owned+additional > r['cap']
            or shutil.disk_usage(root).free-additional < r['minimum_free']):
        raise RuntimeError('Products storage reservation/free-space limit exceeded')


def save_checkpoint(path, state):
    from .storage_budget import tensor_storage_bytes
    from .filesystem_paths import windows_io_path
    allocation_guard(path, tensor_storage_bytes(state)+2**20)
    temporary = unique_temporary_path(path)
    def save():
        with open(windows_io_path(temporary), 'wb') as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
    retry_windows_io(save, temporary)
    atomic_replace(temporary, path)


def canonical_blocks(blocks):
    result = []
    for block in blocks:
        if block.idtype != torch.int64 or not block.is_block:
            raise ValueError('Native int64 sampled block required')
        src, dst = block.srcdata[dgl.NID], block.dstdata[dgl.NID]
        if not torch.equal(src[:len(dst)], dst):
            raise ValueError('Destination/source prefix mismatch')
        u, v = block.edges(order='eid')
        keep = u != v
        loops = torch.arange(len(dst), dtype=torch.int64)
        b = dgl.create_block((torch.cat([u[keep], loops]), torch.cat([v[keep], loops])),
            num_src_nodes=len(src), num_dst_nodes=len(dst), idtype=torch.int64)
        b.srcdata[dgl.NID], b.dstdata[dgl.NID] = src, dst
        result.append(b)
    if len(result) != 2 or not torch.equal(result[0].dstdata[dgl.NID], result[1].srcdata[dgl.NID]):
        raise ValueError('Two-layer feature alignment changed')
    return result


def counter_keys(ids, draw, layer, seed):
    """Stateless SplitMix64 keys; intentional uint64 wrapping, no mutable DGL RNG."""
    with np.errstate(over='ignore'):
        x=np.asarray(ids,dtype=np.uint64)^np.uint64(seed)
        x=x+np.uint64(draw+1)*np.uint64(0x9E3779B97F4A7C15)+np.uint64(layer+1)*np.uint64(0xD1B54A32D192ED03)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        return x^(x>>np.uint64(31))


def selected_offsets(targets,degree,fanout,layer,seed):
    """Floyd edge-offset selection: distinct occurrences, all edges at low degree.

    Counter keys modulo the bound have negligible finite64-bit modulo bias;
    this is explicitly a deterministic engineering sampler, not author PPR.
    """
    chosen=np.tile(np.arange(fanout,dtype=np.int64),(len(targets),1))
    large=np.flatnonzero(degree>fanout)
    if len(large):
        sample=np.full((len(large),fanout),-1,dtype=np.int64)
        for draw in range(fanout):
            bound=degree[large]-fanout+draw+1
            proposed=(counter_keys(targets[large],draw,layer,seed)%bound.astype(np.uint64)).astype(np.int64)
            exists=(sample[:,:draw]==proposed[:,None]).any(1)
            sample[:,draw]=np.where(exists,bound-1,proposed)
        chosen[large]=sample
    return chosen,np.arange(fanout)[None,:]<np.minimum(degree,fanout)[:,None]


def sample_blocks(graph, targets, plan, start, phase):
    # start/phase are cursor metadata only. Choices depend on global ID/layer,
    # so restarting or changing target grouping cannot change the sampled graph.
    if isinstance(graph,MappedCsc):
        ptr,neighbors=graph.indptr,graph.indices
    else:
        if graph.idtype!=torch.int64 or graph.device.type!='cpu':
            raise ValueError('Native CPU int64 fixture graph or frozen mapped CSC required')
        ptr,neighbors,_=graph.adj_tensors('csc')
        ptr,neighbors=ptr.numpy(),neighbors.numpy()
    current=np.array(targets,dtype=np.int64,copy=True)
    if len(np.unique(current))!=len(current) or current.min()<0 or current.max()>=graph.num_nodes():
        raise ValueError('Distinct valid official target IDs required')
    blocks=[]
    for layer in [1,0]:
        fanout=plan['inference']['fanouts'][layer]
        degree=ptr[current+1].astype(np.int64)-ptr[current].astype(np.int64)
        offsets,valid=selected_offsets(current,degree,fanout,layer,plan['inference']['sampling_seed'])
        positions=(ptr[current,None]+offsets)[valid]
        source=neighbors[positions]
        destination=np.broadcast_to(np.arange(len(current))[:,None],valid.shape)[valid]
        # A target prefix is required by the unchanged author's DGL GATConv.
        union=np.unique(np.concatenate([current,source]))
        membership=np.zeros(len(union),dtype=bool)
        membership[np.searchsorted(union,current)]=True
        inputs=np.concatenate([current,union[~membership]])
        lookup=np.empty(len(union),dtype=np.int64)
        lookup[np.searchsorted(union,inputs)]=np.arange(len(inputs))
        u=lookup[np.searchsorted(union,source)]
        keep=u!=destination
        loops=np.arange(len(current),dtype=np.int64)
        b=dgl.create_block((torch.from_numpy(np.concatenate([u[keep],loops])),
            torch.from_numpy(np.concatenate([destination[keep],loops]))),
            num_src_nodes=len(inputs),num_dst_nodes=len(current),idtype=torch.int64)
        b.srcdata[dgl.NID]=torch.from_numpy(inputs)
        b.dstdata[dgl.NID]=torch.from_numpy(current)
        blocks.insert(0,b)
        current=inputs
    return torch.from_numpy(current),blocks


@torch.no_grad()
def forward_blocks(model, data, inputs, blocks, device='cuda'):
    text, vision = data.modality_batch(inputs)
    h1, h2 = text.to(device), vision.to(device)
    for i, block in enumerate(blocks):
        block = block.to(device)
        h1 = F.gelu(model.gat_t5vit.norms[i](model.gat_t5vit.gat_layers[i](block, h1).flatten(1)))
        h2 = F.gelu(model.gat_clip.norms[i](model.gat_clip.gat_layers[i](block, h2).flatten(1)))
    gate = model.fuse_gate(torch.cat([h1, h2], -1))
    return model.fuse_mlp(torch.cat([gate*h1+(1-gate)*h2, h1, h2], -1))


@torch.no_grad()
def encode(model, data, ids, plan, start, phase):
    inputs, blocks = sample_blocks(data.graph, ids, plan, start, phase)
    z = forward_blocks(model, data, inputs, blocks)
    if z.shape != (len(ids), 1024) or z.dtype != torch.float32 or not torch.isfinite(z).all():
        raise FloatingPointError('Full1024-wide finite FP32 CUDA output required')
    return z


class PackedLinearHeads(torch.nn.Module):
    """Independent initialization and parameter rows; one shared matrix multiply."""
    def __init__(self, width, classes, seeds):
        super().__init__()
        weights, biases = [], []
        with torch.random.fork_rng(devices=[]):
            for seed in seeds:
                torch.default_generator.manual_seed(seed)
                head = torch.nn.Linear(width, classes)
                weights.append(head.weight.detach().clone())
                biases.append(head.bias.detach().clone())
        self.weight = torch.nn.Parameter(torch.cat(weights))
        self.bias = torch.nn.Parameter(torch.cat(biases))
        self.classes, self.count = classes, len(seeds)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias).reshape(len(x), self.count, self.classes)


def packed_losses(logits, labels):
    n, s, c = logits.shape
    labels = labels[:, None].expand(n, s).reshape(-1)
    return F.cross_entropy(logits.reshape(n*s, c), labels, reduction='none').reshape(n, s).mean(0)


def metadata(plan):
    from .prepared_data import prepared_directory
    directory = prepared_directory('ogbn-products')
    manifest = read(directory/'manifest.json')
    values = {}
    for key in ['train', 'val', 'test']:
        path = directory/(key+'_indices.npy')
        if fingerprint(path) != manifest['files'][path.name]:
            raise ValueError('Official split changed')
        values[key] = np.load(path, mmap_mode='r')
    path = directory/'labels.npy'
    if fingerprint(path) != manifest['files'][path.name]:
        raise ValueError('Official labels changed')
    labels = np.load(path, mmap_mode='r').reshape(-1)
    if ([len(values[k]) for k in ['train','val','test']] != [196615,39323,2213091]
            or len(labels) != 2449029 or int(labels.min()) != 0 or int(labels.max()) != 46):
        raise ValueError('Complete official Products splits/classes required')
    return values, labels


def load(plan):
    from .prepared_data import load_prepared
    from .train import make_model
    data = load_prepared('ogbn-products', build_graph=False)
    path=data.features.filename
    data.features._mmap.close()
    data.features=ReopenedNpyFeatures(path)
    store = read(ROOT/'configs/storage.json')
    index = Path(store['embedding_cache_root'])/'structural_indexes/products_csc32_v1'
    certificate = read(index/'manifest.json')
    if (not certificate['complete'] or certificate['identity']['nodes'] != 2449029
            or certificate['identity']['source_sha256'] != data.audit['view_files_sha256']['edge_index.npy']):
        raise ValueError('Verified complete structural index required')
    for name, sha in certificate['files'].items():
        if fingerprint(index/name) != sha:
            raise ValueError('Frozen CSC changed')
    # Sampling never needs a full DGL graph or its global edge-ID array.
    # The same CSC edge order feeds unchanged int64 miniblocks on CUDA.
    data.graph = MappedCsc(index)
    encoder = Path(read(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    if fingerprint(encoder) != plan['encoder_sha256']:
        raise ValueError('Original frozen encoder changed')
    saved = torch.load(str(encoder), map_location='cpu', weights_only=False, mmap=True)
    # All parameters are immediately replaced by the complete saved state.
    # Initialize directly on CUDA instead of retaining a redundant CPU model.
    with torch.device('cuda'):
        model = make_model(saved['config'])
    model.load_state_dict(saved['model'])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    cache_plan=plan.get('input_cache')
    if cache_plan:
        torch.cuda.empty_cache()
        size=cache_plan['maximum_bytes']
        if torch.cuda.mem_get_info()[0]<size+cache_plan['cuda_reserve_bytes']:
            raise TrainingPaused('Exact FP32 input-cache CUDA reserve unavailable; no precision/width reduction')
        started=time.perf_counter()
        ids=cache_node_ids(data.graph,size//(512*4))
        cache=ExactFeatureCache(data.features,ids)
        cold=np.flatnonzero(cache.lookup<0)
        probe=np.unique(np.concatenate([ids[:2],cold[:2]]))
        cached,_=cache.modality_batch(torch.from_numpy(probe))
        torch.testing.assert_close(cached.cpu(),torch.from_numpy(data.features[probe]),rtol=0,atol=0)
        data.modality_batch=cache.modality_batch
        data.input_cache=cache
        data.input_cache_upload_seconds=time.perf_counter()-started
    return model, data, saved['config'], encoder


def experiment_config(config, plan, stage):
    config = copy.deepcopy(config)
    config.update(name='products_fast_'+stage, seed=0, storage=read(ROOT/'configs/storage.json'))
    config['evaluation'] = dict(dataset='ogbn-products', plan=plan, stage=stage, test_used_for_selection=False)
    config['deviations'] = list(config.get('deviations', []))+[
        'Time-constrained inference approximation: fixed fanouts='+str(plan['inference']['fanouts'])+'; native int64 blocks and one self-loop per destination; NOT author-confirmed inference.',
        'All official train/val/test IDs and1024-wide FP32 retained; frozen author encoder and pretraining PPR unchanged.',
        'Full CSC structural IDs are read-only int32 mmap; only sampled DGL blocks use int64. No full-graph cast or global edge-ID allocation.',
        'Ten independent linear heads packed into shared GEMM; complete test encoded once after validation-only head selection.']
    return config


def resource_guard(plan):
    snapshot=memory_snapshot()
    if (snapshot['available_RAM_bytes'] < plan['budget']['minimum_free_RAM_bytes']
            or snapshot['process_private_bytes'] > snapshot['total_RAM_bytes']*plan['budget']['maximum_private_RAM_fraction']):
        atomic_write_json(OUT/'memory_boundary.json',dict(snapshot,thresholds=plan['budget'],committed_work_preserved=True))
        raise TrainingPaused('Products RAM boundary: available=%.2fGiB private=%.2fGiB; committed rows/checkpoint retained'%
            (snapshot['available_RAM_bytes']/2**30,snapshot['process_private_bytes']/2**30))


def calibration(plan):
    model, data, config, encoder = load(plan)
    e = ProductsExperiment(experiment_config(config, plan, 'calibration'), data.audit, encoder)
    try:
        atomic_write_json(e.directory/'memory_after_load.json',dict(memory_snapshot(),
            graph_storage='read_only_compact_CSC_int32; sampled_blocks_int64',structural_mapped_bytes=data.graph.structural_bytes))
        e.logger.info('Mapped CSC only: %.1fMiB structural arrays; no full DGL graph/edge-ID expansion; RAM %s',
            data.graph.structural_bytes/2**20,json.dumps(memory_snapshot()))
        if hasattr(data,'input_cache'):e.logger.info('Exact FP32 input cache: %s upload_seconds=%.2f',json.dumps(data.input_cache.stats()),data.input_cache_upload_seconds)
        # Full fanout on a small graph must reproduce the unmodified author forward.
        from .prepared_data import PreparedData
        seed_everything(713)
        small = dgl.graph(([0,1,1,2,3],[1,0,2,3,3]), num_nodes=6).long()
        features = np.random.default_rng(713).normal(size=(6,512)).astype(np.float32)
        fixture = PreparedData(small, features, None, {}, {})
        full = dgl.add_self_loop(dgl.remove_self_loop(small))
        inputs, targets, blocks = dgl.dataloading.MultiLayerFullNeighborSampler(2).sample_blocks(full, torch.arange(6))
        actual = forward_blocks(model, fixture, inputs, canonical_blocks(blocks))
        with torch.no_grad():
            x = torch.from_numpy(features).cuda()
            expected = model(small.to('cuda'), x, x, getz3=True, compute_loss=False)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        # Tiny, explicitly synthetic CUDA fixture: independent-head gradients
        # and the same packed Adam updates before/after a serialized restart.
        fx,fy=torch.randn(11,8,device='cuda'),torch.arange(11,device='cuda')%3
        def fixture_steps(steps,parent=None):
            h=PackedLinearHeads(8,3,[0,1,2]).cuda()
            opt=torch.optim.Adam(h.parameters(),lr=.05)
            if parent:
                old=torch.load(str(parent),map_location='cpu',weights_only=False)
                h.load_state_dict(old['heads']);opt.load_state_dict(old['optimizer']);restore_rng(old['rng'])
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                packed_losses(h(fx),fy).sum().backward()
                opt.step()
            return h,opt
        continuous,co=fixture_steps(3)
        paused,po=fixture_steps(1)
        fixture_path=e.checkpoint_directory/'packed_fixture_resume.pt'
        save_checkpoint(fixture_path,dict(heads=paused.state_dict(),optimizer=po.state_dict(),rng=rng_state(),fixture_only_not_paper=True))
        resumed,ro=fixture_steps(2,fixture_path)
        for key,value in continuous.state_dict().items():
            torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
        for i,state in co.state_dict()['state'].items():
            for key,value in state.items():
                torch.testing.assert_close(value,ro.state_dict()['state'][i][key],rtol=0,atol=0)
        packed=PackedLinearHeads(8,3,[0,1,2]).cuda()
        packed_losses(packed(fx),fy).sum().backward()
        for i in range(3):
            head=torch.nn.Linear(8,3).cuda()
            head.load_state_dict(dict(weight=packed.weight[i*3:(i+1)*3].detach(),bias=packed.bias[i*3:(i+1)*3].detach()))
            F.cross_entropy(head(fx),fy).backward()
            torch.testing.assert_close(head.weight.grad,packed.weight.grad[i*3:(i+1)*3],rtol=2e-5,atol=2e-7)
        # All probe IDs are label-independent. No test scoring happens here.
        ids = np.random.default_rng(217).choice(plan['nodes'], plan['budget']['calibration_nodes'], replace=False)
        batch, rates, durations = plan['inference']['batch_size'], [], []
        first = ids[:batch]
        ia,ba=sample_blocks(data.graph,first,plan,0,0)
        ib,bb=sample_blocks(data.graph,first,plan,0,0)
        torch.testing.assert_close(ia,ib,rtol=0,atol=0)
        for left,right in zip(ba,bb):
            for key in ['srcdata','dstdata']:
                torch.testing.assert_close(getattr(left,key)[dgl.NID],getattr(right,key)[dgl.NID],rtol=0,atol=0)
            for aedge,bedge in zip(left.edges(order='eid'),right.edges(order='eid')):
                torch.testing.assert_close(aedge,bedge,rtol=0,atol=0)
        a = encode(model, data, first, plan, 0, 0)
        b = encode(model, data, first, plan, 0, 0)
        # DGL CUDA scatter reductions are not promised bitwise deterministic.
        # Topology is exact; repeated FP32 neural output uses the author tolerance.
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        replay_difference=float((a-b).abs().max())
        del a, b
        for start in range(0, len(ids), batch):
            resource_guard(plan)
            pause_if_requested()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z = encode(model, data, ids[start:start+batch], plan, start, 0)
            z.cpu().numpy()  # Include the actual feature write's device transfer.
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begin
            rates.append(len(z)/elapsed)
            durations.append(elapsed)
            e.logger.info('CALIBRATION rows=%d/%d batch_nodes_per_second=%.2f', start+len(z), len(ids), rates[-1])
        rate = min(len(ids)/sum(durations), float(np.percentile(rates, 25)))
        cache = Path(read(ROOT/'configs/storage.json')['embedding_cache_root'])/plan['encoder_sha256'][:20]/'ogbn-products'
        old = read(cache/'inference_state.json')
        if old['identity']['checkpoint_sha256'] != plan['encoder_sha256'] or old['identity']['data']!=data.audit or old['cursor'] < 16:
            raise ValueError('Original exact prefix identity unavailable')
        ref_ids = np.linspace(0, old['cursor']-1, 16, dtype=np.int64)
        reference = np.load(cache/'fused_embeddings.npy', mmap_mode='r')
        digest=hashlib.sha256()
        with (cache/'fused_embeddings.npy').open('rb') as f:
            f.seek(reference.offset)
            left=old['cursor']*1024*4
            while left:
                value=f.read(min(left,8*2**20))
                if not value:
                    raise ValueError('Truncated exact reference prefix')
                digest.update(value)
                left-=len(value)
        if digest.hexdigest()!=old['prefix_sha256']:
            raise ValueError('Original exact prefix checksum changed')
        sampled = encode(model, data, ref_ids, plan, 0, 3).cpu().numpy()
        ref = np.array(reference[ref_ids], copy=True)
        cosine = np.sum(ref*sampled, 1)/(np.linalg.norm(ref,axis=1)*np.linalg.norm(sampled,axis=1)).clip(1e-12)
        estimate = estimate_seconds(rate, plan)
        passed = bool(estimate <= plan['budget']['maximum_active_seconds'] and float(cosine.mean()) >= plan['acceptance']['minimum_mean_prefix_cosine'])
        summary = dict(passed=passed, actual_torch_and_DGL_CUDA=True, small_author_forward_allclose=True,
            synthetic_packed_head_CUDA_resume_bitexact=True,packed_gradients_match_independent_heads=True,
            fixed_sampled_topology_bitexact=True,repeated_FP32_forward_allclose=True,max_replay_abs_difference=replay_difference,
            calibration_nodes=len(ids), conservative_nodes_per_second=rate,
            estimated_total_active_seconds=estimate, prefix_mean_cosine=float(cosine.mean()),
            prefix_min_cosine=float(cosine.min()), peak_CUDA_bytes=torch.cuda.max_memory_allocated(),
            test_evaluations=0, test_accuracy=None, inference_is_approximation=True)
        if hasattr(data,'input_cache'):summary.update(input_cache=data.input_cache.stats(),input_cache_upload_seconds=data.input_cache_upload_seconds,input_cache_CUDA_hit_and_miss_bitexact=True)
        atomic_write_json(e.directory/'summary.json', summary)
        e.metric(dict(step=1, nodes_per_second=rate, estimated_seconds=estimate, prefix_cosine=float(cosine.mean()), passed=int(passed)))
        e.status('completed_products_fast_calibration')
    except BaseException as error:
        e.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error))
        raise
    finally:
        e.close()


def features(plan, resume=None):
    model, data, config, encoder = load(plan)
    splits, labels = metadata(plan)
    ids = np.concatenate([splits['train'], splits['val']])
    e = ProductsExperiment(experiment_config(config, plan, 'features'), data.audit, resume or encoder)
    try:
        cursor = RowCursor(e.checkpoint_directory/'feature_cursor.json', dict(plan=plan, data=data.audit, phase=1),
            (len(ids),1024), 'float32', resume, allocation_guard)
        batch, last = plan['inference']['batch_size'], time.monotonic()
        while cursor.cursor < len(ids):
            resource_guard(plan)
            pause_if_requested()
            start = cursor.cursor
            z = encode(model, data, ids[start:start+batch], plan, start, 1).cpu().numpy()
            cursor.commit(start, z)
            if time.monotonic()-last > 30 or cursor.cursor == len(ids):
                e.logger.info('FEATURE_ROWS %d/%d; complete official train+validation', cursor.cursor, len(ids))
                e.metric(dict(step=cursor.cursor, feature_rows=cursor.cursor, total_rows=len(ids)))
                last = time.monotonic()
        matrix = cursor.complete_values()
        atomic_write_json(e.directory/'summary.json', dict(features_path=str(cursor.values_path),
            features_sha256=fingerprint(cursor.values_path), shape=list(matrix.shape), full_official_train_and_validation=True,
            cursor_path=str(cursor.path), test_evaluations=0, data_audit=data.audit))
        e.status('completed_products_fast_features')
    except BaseException as error:
        e.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error))
        raise
    finally:
        e.close()


def fit_heads(plan, feature_run, resume=None):
    from .prepared_data import prepared_directory
    feature_run = Path(feature_run)
    source = read(feature_run/'summary.json')
    if fingerprint(source['features_path']) != source['features_sha256']:
        raise ValueError('Completed sampled train/validation features changed')
    splits, labels = metadata(plan)
    matrix = np.load(source['features_path'], mmap_mode='r')
    config = experiment_config(read(feature_run/'config.json'), plan, 'heads')
    config['evaluation']['feature_summary_sha256'] = fingerprint(feature_run/'summary.json')
    e = ProductsExperiment(config, source['data_audit'], resume or Path(source['cursor_path']))
    try:
        seed_everything(713)
        x=torch.empty(matrix.shape,dtype=torch.float32,device='cuda')
        for start in range(0,len(matrix),8192):
            x[start:start+8192]=torch.from_numpy(np.array(matrix[start:start+8192],copy=True)).cuda()
        n = len(splits['train'])
        train, valid = x[:n], x[n:]
        y = torch.from_numpy(np.array(labels[splits['train']], dtype=np.int64)).cuda()
        vy = torch.from_numpy(np.array(labels[splits['val']], dtype=np.int64)).cuda()
        heads = PackedLinearHeads(1024,47,plan['head']['seeds']).cuda()
        optimizer = torch.optim.Adam(heads.parameters(), lr=plan['head']['lr'])
        best = np.full(10,-1.)
        selected, stale = np.zeros(10,np.int64), np.zeros(10,np.int64)
        best_weight, best_bias = heads.weight.detach().cpu().clone(), heads.bias.detach().cpu().clone()
        history, first = [], 1
        if resume:
            old = torch.load(str(resume), map_location='cpu', weights_only=False)
            if old['schema'] != 'came_products_packed_heads_v1' or old['config'] != config:
                raise ValueError('Packed head resume identity changed')
            heads.load_state_dict(old['heads'])
            optimizer.load_state_dict(old['optimizer'])
            restore_rng(old['rng'])
            best, selected, stale, best_weight, best_bias, history, first = [old[k] for k in
                ['best','selected','stale','best_weight','best_bias','history','next_epoch']]
            for row in history:
                e.metric(row)
        def save(next_epoch):
            save_checkpoint(e.checkpoint_directory/'heads_last.pt', dict(schema='came_products_packed_heads_v1',
                heads=heads.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(), best=best,
                selected=selected, stale=stale, best_weight=best_weight, best_bias=best_bias,
                history=history, next_epoch=next_epoch, config=config))
        last_epoch=first-1
        for epoch in range(first, plan['head']['max_epochs']+1):
            active = stale < plan['head']['patience']
            if not active.any():
                break
            if pause_requested_local():
                save(epoch)
                raise TrainingPaused('Packed heads saved before next complete full-batch update')
            heads.train()
            optimizer.zero_grad(set_to_none=True)
            losses = packed_losses(heads(train),y)
            if not torch.isfinite(losses).all():
                raise FloatingPointError('Nonfinite classifier objective')
            losses[torch.as_tensor(active,device='cuda')].sum().backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters()):
                raise FloatingPointError('Nonfinite independent-head gradients')
            frozen = torch.as_tensor(np.repeat(~active,47), device='cuda')
            for param in heads.parameters():
                state = optimizer.state.get(param, {})
                for key in ['exp_avg','exp_avg_sq']:
                    if key in state:
                        state[key][frozen] = 0
            optimizer.step()
            heads.eval()
            with torch.no_grad():
                logits=heads(valid)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite validation logits')
                scores = (logits.argmax(2)==vy[:,None]).sum(0).cpu().numpy().astype(np.float64)/len(vy)
            for i in range(10):
                if not active[i]:
                    continue
                if scores[i] > best[i]:
                    best[i], selected[i], stale[i] = scores[i], epoch, 0
                    sl = slice(i*47,(i+1)*47)
                    best_weight[sl] = heads.weight[sl].detach().cpu()
                    best_bias[sl] = heads.bias[sl].detach().cpu()
                else:
                    stale[i] += 1
            row = dict(step=epoch, train_loss=float(losses.detach().mean()),
                validation_accuracy_mean=float(scores.mean()), active_heads=int(active.sum()))
            loss_values=losses.detach().cpu().numpy()
            for i in range(10):
                row.update({'train_loss_seed%d'%i:float(loss_values[i]),
                    'validation_accuracy_seed%d'%i:float(scores[i]),
                    'selected_epoch_seed%d'%i:int(selected[i]),'active_seed%d'%i:int(active[i])})
            history.append(row)
            last_epoch=epoch
            e.metric(row)
            if epoch == 1 or epoch % 25 == 0:
                e.logger.info('PACKED_HEADS epoch=%d/2000 active=%d validation_mean=%.6f; ten independent seeds', epoch, active.sum(), scores.mean())
            if epoch == 1 or epoch % 25 == 0 or pause_requested_local():
                save(epoch+1)
        save(last_epoch+1)
        head_path = e.checkpoint_directory/'heads_best.pt'
        save_checkpoint(head_path, dict(weight=best_weight, bias=best_bias, selected=selected, validation_accuracy=best, seeds=plan['head']['seeds']))
        heads.load_state_dict(dict(weight=best_weight,bias=best_bias))
        with torch.no_grad():
            predictions = heads(valid).argmax(2).cpu().numpy()
        np.testing.assert_array_equal(np.mean(predictions==np.array(labels[splits['val']])[:,None],0),best)
        np.savez_compressed(e.directory/'validation_predictions.npz',prediction=predictions,node_ids=splits['val'],labels=labels[splits['val']])
        atomic_write_json(e.directory/'summary.json', dict(classifier_checkpoint=str(head_path), classifier_sha256=fingerprint(head_path),
            feature_run_directory=str(feature_run), validation_accuracy=best.tolist(), selected_epoch=selected.tolist(),
            seeds=plan['head']['seeds'], test_evaluations=0, completed_full_batch_epochs=last_epoch,
            official_train_nodes=196615,official_validation_nodes=39323,head_epochs_upper_bound=2000, per_head_patience=100))
        draw_history(history,e.directory/'figures/heads_training.png')
        e.status('completed_products_fast_heads')
    except BaseException as error:
        e.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error))
        raise
    finally:
        e.close()


def pause_requested_local():
    from .pause_control import pause_requested
    return pause_requested()


def draw_history(history,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['step'] for r in history],[r['train_loss'] for r in history])
    axes[0].set(xlabel='Full-batch epoch',ylabel='Mean training cross-entropy')
    axes[1].plot([r['step'] for r in history],[100*r['validation_accuracy_mean'] for r in history])
    axes[1].set(xlabel='Full-batch epoch',ylabel='Mean validation accuracy (%)')
    fig.suptitle('Products: ten independent heads; sampled frozen representations; test held out')
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def test_heads(plan, head_run, lock_path, resume=None):
    head_run=Path(head_run)
    summary=read(head_run/'summary.json')
    lock=read(lock_path)
    if (summary['test_evaluations']!=0 or not lock['locked_before_test'] or lock['test_used_for_selection']
            or fingerprint(head_run/'summary.json')!=lock['head_summary_sha256']
            or fingerprint(summary['classifier_checkpoint'])!=lock['classifier_sha256']):
        raise ValueError('All ten validation-selected heads must be frozen BEFORE test')
    model,data,config,encoder=load(plan)
    splits,labels=metadata(plan)
    saved=torch.load(summary['classifier_checkpoint'],map_location='cpu',weights_only=False)
    heads=PackedLinearHeads(1024,47,plan['head']['seeds']).cuda()
    heads.load_state_dict(dict(weight=saved['weight'],bias=saved['bias']))
    heads.eval()
    config=experiment_config(config,plan,'test')
    config['evaluation']['selection_lock_sha256']=fingerprint(lock_path)
    e=ProductsExperiment(config,data.audit,resume or Path(summary['classifier_checkpoint']))
    try:
        feature_source=read(Path(summary['feature_run_directory'])/'summary.json')
        if fingerprint(feature_source['features_path'])!=feature_source['features_sha256']:
            raise ValueError('Complete validation feature source changed')
        feature_matrix=np.load(feature_source['features_path'],mmap_mode='r')
        with torch.no_grad():
            vx=torch.from_numpy(np.array(feature_matrix[196615:],copy=True)).cuda()
            validation_replay=heads(vx).argmax(2).cpu().numpy()
            del vx
        with np.load(head_run/'validation_predictions.npz') as old:
            np.testing.assert_array_equal(validation_replay,old['prediction'])
        actual_valid=np.mean(validation_replay==np.array(labels[splits['val']])[:,None],0)
        np.testing.assert_array_equal(actual_valid,summary['validation_accuracy'])
        cursor=RowCursor(e.checkpoint_directory/'predictions_cursor.json',dict(plan=plan,lock_sha256=fingerprint(lock_path),phase=2),
            (2213091,10),'uint8',resume,allocation_guard)
        batch,last=plan['inference']['batch_size'],time.monotonic()
        while cursor.cursor<2213091:
            resource_guard(plan)
            pause_if_requested()
            start=cursor.cursor
            with torch.no_grad():
                z=encode(model,data,splits['test'][start:start+batch],plan,start,2)
                logits=heads(z)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite official test logits')
                pred=logits.argmax(2).cpu().numpy().astype(np.uint8)
            cursor.commit(start,pred)
            if time.monotonic()-last>30 or cursor.cursor==2213091:
                e.logger.info('FULL_OFFICIAL_TEST_ROWS %d/2213091 heads=10; no interim test-score selection',cursor.cursor)
                e.metric(dict(step=cursor.cursor,official_test_rows=cursor.cursor,head_count=10,seed=None,test_accuracy=None))
                last=time.monotonic()
        predictions=cursor.complete_values()
        test_labels=np.array(labels[splits['test']],copy=True)
        scores=np.mean(predictions==test_labels[:,None],axis=0)
        np.savez_compressed(e.directory/'all_seeds_test_predictions.npz',prediction=predictions,node_ids=splits['test'],labels=test_labels,seeds=plan['head']['seeds'])
        result=dict(dataset='ogbn-products',seeds=plan['head']['seeds'],test_accuracy=scores.tolist(),
            mean_pct=float(scores.mean()*100),sample_sd_pp=float(scores.std(ddof=1)*100),paper_pct=plan['paper_accuracy_pct'],
            gap_pp=float(scores.mean()*100-plan['paper_accuracy_pct']),official_test_nodes=2213091,
            test_evaluations_per_seed=1,inference_is_sampled_approximation=True,paper_reproduction_complete=False,
            head_run_directory=str(head_run),selection_lock_sha256=fingerprint(lock_path),complete_cursor=str(cursor.path))
        atomic_write_json(e.directory/'summary.json',result)
        for i,score in enumerate(scores):
            e.metric(dict(step=2213092+i,official_test_rows=2213091,head_count=10,seed=i,test_accuracy=float(score)))
        e.logger.info('COMPLETE_PRODUCTS ten_seeds mean=%.8f%% sample_sd=%.8fpp paper=%.2f%% gap=%+.8fpp',
            result['mean_pct'],result['sample_sd_pp'],result['paper_pct'],result['gap_pp'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(plan['head']['seeds'],100*scores,'o-',label='Full official test; sampled frozen inference')
        ax.axhline(plan['paper_accuracy_pct'],color='black',ls='--',label='Paper81.84')
        ax.set(xlabel='Downstream seed',ylabel='Official test accuracy (%)',title='Products:10/10 independent heads')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(e.directory/'figures/products_comparison.png',dpi=180)
        plt.close(fig)
        e.status('completed_products_fast_ten_official_tests')
    except BaseException as error:
        e.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error))
        raise
    finally:
        e.close()
