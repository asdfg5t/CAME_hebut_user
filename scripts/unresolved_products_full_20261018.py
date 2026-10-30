"""Exact Products continuation using pre-certified compact incoming CSC storage."""
import copy
import gc
import hashlib
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_products_single_validation import next_cursor


def resource_guard():
    import psutil
    from unresolved_backend_20261018 import read, PROFILE
    from came_repro.pause_control import TrainingPaused
    p=read(PROFILE)['products'];vm=psutil.virtual_memory();m=psutil.Process().memory_info()
    if getattr(m,'private',m.rss)>p['maximum_process_memory_fraction']*vm.total or vm.available<p['minimum_available_RAM_bytes']:
        raise TrainingPaused('RAM guard: saved embedding cursor retained; no CPU computation fallback')


def exact_inference(model,data,cache,model_hash,experiment,batch):
    import dgl
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file,atomic_write_json
    from came_repro.author_loader import fingerprint
    from came_repro.stream_inference import batch_full_neighbor_embeddings
    from came_repro.pause_control import pause_if_requested
    from came_repro.storage_budget import guard_allocation
    cache.mkdir(parents=True,exist_ok=True);target=cache/'fused_embeddings.npy';state_path=cache/'inference_state.json'
    n=data.graph.num_nodes();identity={'checkpoint_sha256':model_hash,'data':data.audit,'hidden_dim':1024,'method':'All inbound neighbors, all encoder layers, eval mode; no PPR approximation','precision':'fp32'}
    state=read_json_file(state_path) if state_path.exists() else {'identity':identity,'cursor':0,'complete':False,'prefix_sha256':hashlib.sha256().hexdigest()}
    if state['identity']!=identity:raise ValueError('Products cache identity changed')
    if state['complete']:
        if fingerprint(target)!=state['output_sha256']:raise ValueError('Completed Products output checksum changed')
        return np.load(target,mmap_mode='r')
    if state['cursor'] and not target.exists():raise ValueError('Saved cursor has no output file')
    if not target.exists():
        guard_allocation(target,n*1024*4+2**20)
        arr=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(n,1024));arr.flush();arr._mmap.close()
    arr=np.load(target,mmap_mode='r')
    if arr.shape!=(n,1024) or arr.dtype!=np.float32:raise ValueError('Full FP32 output shape required')
    offset=arr.offset;arr._mmap.close();prefix=hashlib.sha256()
    with target.open('rb') as source:
        source.seek(offset);remaining=state['cursor']*1024*4
        while remaining:
            block=source.read(min(8*2**20,remaining))
            if not block:raise ValueError('Truncated saved Products prefix')
            prefix.update(block);remaining-=len(block)
    if prefix.hexdigest()!=state['prefix_sha256']:raise ValueError('Saved Products prefix checksum changed')
    graph=data.graph  # CSC certificate already implements exact author loop semantics.
    sampler=dgl.dataloading.MultiLayerFullNeighborSampler(2);model.eval();started=time.perf_counter();start_cursor=state['cursor'];last_log=started
    for parameter in model.parameters():parameter.requires_grad_(False)
    experiment.logger.info('Products exact inference cursor=%d/%d, batch=%d; prefix checksum verified',state['cursor'],n,batch)
    with target.open('r+b') as destination,torch.no_grad():
        while state['cursor']<n:
            resource_guard();pause_if_requested();cursor=state['cursor'];stop=min(cursor+batch,n);targets=torch.arange(cursor,stop,dtype=graph.idtype)
            inputs=blocks=fused=None;oom=False
            try:
                inputs,output,blocks=sampler.sample_blocks(graph,targets)
                fused=batch_full_neighbor_embeddings(model,data,blocks,inputs)
                if fused.shape!=(stop-cursor,1024) or fused.dtype!=torch.float32 or not torch.isfinite(fused).all():raise FloatingPointError('Nonfinite, non-FP32 or wrong-width Products encoding')
                rows=fused.cpu().numpy()
            except torch.cuda.OutOfMemoryError:oom=True
            if oom:
                del inputs,blocks,fused;gc.collect();torch.cuda.empty_cache()
                if batch==1:raise RuntimeError('Exact single-target inference cannot fit; requires separate exact capacity redesign')
                batch=max(1,batch//2);experiment.logger.warning('CUDA OOM: retry SAME uncommitted cursor=%d at batch=%d',cursor,batch);continue
            new_cursor=next_cursor(output.numpy(),cursor,n);blob=rows.tobytes(order='C')
            destination.seek(offset+cursor*1024*4);written=destination.write(blob)
            if written!=len(blob):raise OSError('Incomplete Products row write')
            destination.flush();os.fsync(destination.fileno());prefix.update(blob)
            state.update(cursor=new_cursor,prefix_sha256=prefix.hexdigest(),last_batch_size=batch)
            atomic_write_json(state_path,state)
            del inputs,blocks,fused,rows,blob
            if time.perf_counter()-last_log>=30 or state['cursor']==n:
                seconds=time.perf_counter()-started;done=state['cursor']-start_cursor
                experiment.logger.info('Products exact nodes=%d/%d active_elapsed=%.1fs throughput=%.2f nodes/s batch=%d',state['cursor'],n,seconds,done/max(seconds,1e-9),batch);last_log=time.perf_counter()
    state.update(complete=True,output_sha256=fingerprint(target));atomic_write_json(state_path,state)
    return np.load(target,mmap_mode='r')


def main(resume=None):
    import numpy as np
    import torch
    from unresolved_backend_20261018 import read, storage, encoder, product_data, OUT, PROFILE
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.experiment import Experiment
    from came_repro.train import make_model
    from came_repro.downstream import node_inputs, train_linear_probe, score_probe
    from came_repro.quality_probe import CachedProbeInput
    from came_repro.pause_control import TrainingPaused
    plan = read(PROFILE); accepted = read(OUT/'products_calibration.json')
    if (not accepted['ready_for_full_start'] or accepted['profile_sha256'] != fingerprint(PROFILE)
            or accepted['index_manifest_sha256'] != fingerprint(Path(storage()['embedding_cache_root'])/'structural_indexes/products_csc32_v1/manifest.json')):
        raise ValueError('Current numerical, memory and throughput acceptance required')
    data, index = product_data()
    if data.audit['nodes'] != 2449029 or len(data.splits['test']) != 2213091:
        raise ValueError('Full official Products graph/test required')
    saved = torch.load(str(encoder()), map_location='cpu', weights_only=False, mmap=True)
    config = copy.deepcopy(saved['config']); config.update(name='products_compact_full', seed=0, storage=storage())
    config['evaluation'] = dict(dataset='ogbn-products', task='node', seed=0, epochs=2000, patience=100,
        probe_batch_size=0, full_test_nodes=2213091, selection_split='official_validation',
        test_used_for_selection=False, profile_sha256=fingerprint(PROFILE),
        compact_index_manifest_sha256=accepted['index_manifest_sha256'])
    config['deviations'] = list(config.get('deviations', []))+[
        'Structural int32 CSC storage only; exact original parallel edges, one self-loop per node and full-neighbor FP32 author model preserved.',
        'Original committed Products embedding prefix verified and retained; continuation numerical acceptance recorded.']
    parent = resume or encoder(); e = Experiment(config, data.audit, parent)
    cache = Path(storage()['embedding_cache_root'])/plan['encoder_sha256'][:20]/'ogbn-products'
    try:
        model = make_model(config); model.load_state_dict(saved['model']); del saved
        embeddings = exact_inference(model, data, cache, plan['encoder_sha256'], e, accepted['selected_batch_size'])
        del model; data.graph = None; gc.collect(); torch.cuda.empty_cache()
        train, valid, test = node_inputs(data, embeddings, 0)
        valid = CachedProbeInput(embeddings, valid.labels, indices=valid.indices); valid.cache_validation(400, e.logger)
        summary = train_linear_probe(train, valid, None, 47, 2000, 100, 0, e, resume=resume, evaluate_test=False)
        summary.update(dataset='ogbn-products', seed=0, encoder_sha256=plan['encoder_sha256'],
            embedding_cache=str(cache), official_train_nodes=len(train), official_validation_nodes=len(valid),
            official_test_nodes=len(test), diagnostic_only=True, paper_reproduction_complete=False)
        atomic_write_json(e.directory/'summary.json', summary); e.status('completed_full_products_validation')
        selected = e.directory; head = e.checkpoint_directory/'probe_best.pt'
    except BaseException as error:
        e.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error)); raise
    finally: e.close()
    # One confirmation from the saved validation-selected head, with a durable lock.
    lock = dict(locked_before_test=True, test_used_for_selection=False, classifier_checkpoint=str(head),
        classifier_sha256=fingerprint(head), validation_run=str(selected),
        validation_summary_sha256=fingerprint(selected/'summary.json'), cache_state_sha256=fingerprint(cache/'inference_state.json'))
    lock_path = OUT/'products_selection_lock.json'
    if lock_path.exists():
        if read(lock_path) != lock: raise ValueError('Products test lock already exists; refuse repeated confirmation')
        if (OUT/'products_test_started.json').exists(): raise ValueError('Test previously started; manual audit required, no automatic retest')
    atomic_write_json(lock_path, lock)
    config = copy.deepcopy(config); config['name'] = 'products_compact_test'
    e = Experiment(config, data.audit, head)
    atomic_write_json(OUT/'products_test_started.json', dict(run=str(e.directory), classifier_sha256=lock['classifier_sha256']))
    try:
        classifier = torch.nn.Linear(1024, 47).cuda()
        classifier.load_state_dict(torch.load(str(head), map_location='cpu', weights_only=False)['classifier'])
        value, _, prediction = score_probe(classifier, valid, save_predictions=True)
        if value != summary['validation_accuracy']: raise ValueError('Saved-head validation changed')
        value, loss, prediction = score_probe(classifier, test, save_predictions=True)
        if len(prediction) != 2213091 or float(np.mean(prediction == test.labels)) != value:
            raise ValueError('Full official test recomputation failed')
        np.savez_compressed(e.directory/'test_predictions.npz', prediction=prediction, labels=test.labels, node_ids=test.indices)
        summary.update(test_accuracy=value, test_loss=loss, test_evaluations=1,
            validation_run_directory=str(selected), selection_lock_sha256=fingerprint(lock_path))
        atomic_write_json(e.directory/'summary.json', summary)
        e.metric(dict(step=1, validation_accuracy=summary['validation_accuracy'], test_accuracy=value))
        e.status('completed_products_compact_official_test')
        atomic_write_json(OUT/'products_complete.json', dict(run=str(e.directory), summary=summary))
    except BaseException as error: e.status('failed', error=str(error)); raise
    finally: e.close()

