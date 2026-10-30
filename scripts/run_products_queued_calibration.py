"""Wait for protocol repairs, then calibrate exact Products inference on the free GPU.

This is resource/numerical acceptance, not a test score or a complete Products run.
No source modules or existing configuration files are changed by this helper.
"""
import argparse
import copy
import datetime
import gc
import json
import sys
import time
from pathlib import Path

if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    import psutil
    import numpy as np
    import torch
    import dgl
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from came_repro.experiment import Experiment, seed_everything, write_json
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.prepared_data import load_prepared
    from came_repro.protocol_repair import check_cuda
    from came_repro.storage_budget import check_budget
    from came_repro.stream_inference import batch_full_neighbor_embeddings
    from came_repro.train import make_model

    path = ROOT/'artifacts/workflow_state/products_capacity_current.json'
    repair_path = ROOT/'artifacts/workflow_state/all_unmet_repair_current.json'
    sources = {str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}
    self_hash = fingerprint(__file__)
    state = {'state':'waiting_for_protocol_repair_GPU_release','diagnostic_only':True,
             'test_evaluations':0,'script_sha256':self_hash,'full_nodes':2449029,'width':1024}
    atomic_write_json(path,state)
    while True:
        repair = read_json_file(repair_path)
        if (ROOT/'artifacts/workflow_state/all_unmet_repair_pause.json').exists() or repair['state'] in ['paused','failed']:
            state.update(state='deferred_prior_task_pause_or_failure');atomic_write_json(path,state);return
        if repair['state'] == 'completed_bounded_graph_text_screen':
            try:
                guard = acquire_gpu_lock(ROOT/'.runtime/all_unmet_repair.lock')
                guard.close();break
            except RuntimeError:
                time.sleep(10);continue
        if not psutil.pid_exists(args.wait_pid):
            state.update(state='deferred_parent_exit_requires_review');atomic_write_json(path,state);return
        time.sleep(15)
    verify_originals()
    for name,digest in sources.items():
        if fingerprint(ROOT/name) != digest:raise ValueError('Queued calibration source identity changed')
    if fingerprint(__file__) != self_hash:raise ValueError('Queued calibration script changed')
    model_path = Path(read_json_file(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    digest = fingerprint(model_path)
    gpu_lock = acquire_gpu_lock()
    experiment = None
    try:
        check_cuda();seed_everything(0);torch.set_num_threads(4)
        budget = check_budget(additional_cache_bytes=2449029*1024*4+128*2**20)
        saved = torch.load(str(model_path),map_location='cpu',weights_only=False,mmap=True)
        config = copy.deepcopy(saved['config']);config.update(name='products_capacity_calibration',seed=0,storage=read_json_file(ROOT/'configs/storage.json'))
        config['evaluation'] = {'dataset':'ogbn-products','stage':'resource_numeric_calibration','full_nodes':2449029,
            'width':1024,'no_test_scoring':True,'source_script_sha256':self_hash,
            'probe_selection':'top4 incoming degree and 12 uniform unlabeled IDs; no labels or test score used',
            'batch_candidates':[1,4,8,16],'full_neighbors':True,'precision':'fp32','CPU_compute_fallback':False}
        state.update(state='calibrating',started_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),budget=budget)
        atomic_write_json(path,state)
        data = load_prepared('ogbn-products')
        if data.graph.num_nodes()!=2449029:raise ValueError('Full official graph required')
        experiment = Experiment(config,dict(data.audit,calibration_script_sha256=self_hash),model_path)
        state['run_directory']=str(experiment.directory);atomic_write_json(path,state)
        model = make_model(config);model.load_state_dict(saved['model']);model.eval();del saved
        # The author removes old self loops and adds one per node. CSC changes storage only.
        graph = model._prep_graph(data.graph)
        data.graph = None;gc.collect()
        graph = graph.formats('csc');data.graph=graph;gc.collect()
        degrees = graph.in_degrees().numpy()
        top = np.argpartition(degrees,-4)[-4:];top=top[np.argsort(degrees[top])[::-1]]
        random = np.random.default_rng(217).choice(len(degrees),12,replace=False)
        probes = np.unique(np.concatenate((top,random)))
        # Keep largest-degree rows first so larger batches test a demanding combined case.
        probes = np.concatenate((top,probes[~np.isin(probes,top)]))
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        reference = None;results=[];safe_batch=None
        for batch in [1,4,8,16]:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.perf_counter();rows=[];maximum_inputs=0;maximum_edges=0
            try:
                for start in range(0,len(probes),batch):
                    targets=torch.tensor(probes[start:start+batch],dtype=torch.long)
                    inputs,output,blocks=sampler.sample_blocks(graph,targets)
                    maximum_inputs=max(maximum_inputs,len(inputs));maximum_edges=max(maximum_edges,sum(b.num_edges() for b in blocks))
                    with torch.no_grad():fused=batch_full_neighbor_embeddings(model,data,blocks,inputs)
                    if not torch.isfinite(fused).all():raise FloatingPointError('Nonfinite Products calibration output')
                    rows.append(fused.cpu().numpy());del blocks,inputs,fused
                torch.cuda.synchronize();values=np.concatenate(rows)
                if reference is None:reference=values.copy()
                np.testing.assert_allclose(reference,values,rtol=2e-5,atol=2e-5)
                result={'batch_size':batch,'passed':True,'sampled_nodes':len(probes),'elapsed_seconds':time.perf_counter()-started,
                    'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'maximum_input_nodes':maximum_inputs,'maximum_block_edges':maximum_edges,
                    'max_abs_difference_vs_batch1':float(np.max(np.abs(reference-values)))}
                results.append(result);safe_batch=batch
                experiment.metric(dict(step=batch,peak_cuda_mib=result['peak_cuda_bytes']/2**20,elapsed_seconds=result['elapsed_seconds'],max_abs_difference=result['max_abs_difference_vs_batch1']))
                experiment.logger.info('Products calibration: %s',result)
            except torch.cuda.OutOfMemoryError as error:
                results.append({'batch_size':batch,'passed':False,'reason':'CUDA OOM','error':str(error)})
                gc.collect();torch.cuda.empty_cache();break
        import psutil
        memory=psutil.Process().memory_info();peak=getattr(memory,'peak_wset',memory.rss)
        passed = safe_batch is not None and peak < int(.85*psutil.virtual_memory().total)
        report={'passed':passed,'selected_batch_size':safe_batch,'records':results,'probe_ids':probes.tolist(),
            'largest_probe_in_degree':int(degrees[top[0]]),'process_peak_RSS_bytes':peak,'full_graph_nodes':2449029,
            'full_graph_edges_with_author_self_loops':graph.num_edges(),'embedding_output_bytes':2449029*1024*4,
            'all_official_test_retained':True,'official_test_scored':False,'test_evaluations':0,'encoder_sha256':digest,
            'script_sha256':self_hash,'sources':sources,'budget':budget,
            'limitation':'stress/uniform samples are capacity evidence, not a guarantee for every batch or a throughput ETA; runtime must preserve cursor and refuse CPU fallback'}
        write_json(experiment.directory/'summary.json',report)
        experiment.status('completed_products_capacity_calibration' if passed else 'capacity_limit_requires_layerwise_review')
        state.update(state='completed_calibration_ready_for_single_seed' if passed else 'capacity_limit_requires_layerwise_review',summary=report)
        atomic_write_json(path,state)
    except BaseException as error:
        state.update(state='failed_calibration',error=str(error));atomic_write_json(path,state)
        if experiment:experiment.status('failed',error=str(error))
        raise
    finally:
        if experiment:experiment.close()
        gpu_lock.close()


if __name__ == '__main__':main()
