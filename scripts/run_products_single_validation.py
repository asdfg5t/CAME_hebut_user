"""Exact, resumable Products CUDA inference and full official one-seed classification."""
import argparse
import copy
import gc
import hashlib
import os
import sys
import time
from pathlib import Path

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def next_cursor(targets,cursor,n):
    import numpy as np
    ids=np.asarray(targets)
    if ids.ndim!=1 or not len(ids) or not np.array_equal(ids,np.arange(cursor,cursor+len(ids))) or cursor+len(ids)>n:
        raise ValueError('Inference must commit exactly the next contiguous full target rows')
    return cursor+len(ids)


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
    graph=model._prep_graph(data.graph);data.graph=None;gc.collect();graph=graph.formats('csc');data.graph=graph
    sampler=dgl.dataloading.MultiLayerFullNeighborSampler(2);model.eval();started=time.perf_counter();start_cursor=state['cursor'];last_log=started
    for parameter in model.parameters():parameter.requires_grad_(False)
    experiment.logger.info('Products exact inference cursor=%d/%d, batch=%d; prefix checksum verified',state['cursor'],n,batch)
    with target.open('r+b') as destination,torch.no_grad():
        while state['cursor']<n:
            pause_if_requested();cursor=state['cursor'];stop=min(cursor+batch,n);targets=torch.arange(cursor,stop,dtype=torch.long)
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


def main():
    import json
    import numpy as np
    import torch
    from came_repro.atomic_io import read_json_file,atomic_write_json
    from came_repro.author_loader import fingerprint,verify_originals
    from came_repro.ide_workflow import acquire_gpu_lock
    from came_repro.experiment import Experiment,write_json,seed_everything
    from came_repro.prepared_data import load_prepared
    from came_repro.protocol_repair import check_cuda
    from came_repro.storage_budget import check_budget
    from came_repro.train import make_model
    from came_repro.downstream import node_inputs,train_linear_probe,score_probe
    from came_repro.quality_probe import CachedProbeInput
    from came_repro.pause_control import TrainingPaused
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume-head',type=Path)
    parser.add_argument('--after-sports',action='store_true',help='Wait for numbered Sports ten-seed completion instead of the historical convergence queue')
    parser.add_argument('--pause-file',type=Path,help='Cooperative pause file of the authorized current diagnostic supervisor')
    args=parser.parse_args()
    profile_path=ROOT/'configs/products_single_seed_20261014.json';profile=read_json_file(profile_path)
    path=ROOT/'artifacts/workflow_state/products_single_seed_current.json'
    original_pause=ROOT/'artifacts/workflow_state/products_single_seed_pause.json'
    pause=args.pause_file or original_pause
    os.environ['CAME_PAUSE_FILE']=str(pause)
    sources={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'src/came_repro').glob('*.py')}
    identity={'profile_sha256':fingerprint(profile_path),'script_sha256':fingerprint(__file__),'sources':sources}
    ledger=read_json_file(path) if path.exists() else {'identity':identity,'state':'waiting_for_convergence_GPU_release','test_evaluations':0,'attempts':[]}
    if ledger['identity']!=identity:raise ValueError('Products single-seed identity changed')
    if ledger['state']=='completed_verified_official_single_seed':return
    atomic_write_json(path,ledger);supervisor=acquire_gpu_lock(ROOT/'.runtime/products_single_seed.lock');lock=None;experiment=None
    try:
        while True:
            if pause.exists() or original_pause.exists():ledger['state']='paused';atomic_write_json(path,ledger);return
            prior=read_json_file(ROOT/('artifacts/workflow_state/verified_job07_sports_delivery_repeats_current.json' if args.after_sports else 'artifacts/workflow_state/convergence_repair_current.json'))
            if prior['state'] in ('failed','paused'):
                ledger['state']='deferred_prior_failure_or_pause';atomic_write_json(path,ledger);return
            expected='completed_delivered_two_lp_repeats' if args.after_sports else 'completed_bounded_convergence_screen'
            if prior['state']==expected:
                if args.after_sports:
                    progress=prior['statistics']['sports-copurchase']
                    if progress['completed']!=10 or not progress['complete_repeat_grid']:raise ValueError('All numbered Sports seeds must be complete before Products')
                try:lock=acquire_gpu_lock();break
                except RuntimeError:pass
            time.sleep(15)
        capacity=read_json_file(ROOT/'artifacts/workflow_state/products_capacity_current.json')
        if capacity['state']!='completed_calibration_ready_for_single_seed' or not capacity['summary']['passed']:raise ValueError('Products capacity and numeric acceptance required')
        for name,digest in sources.items():
            if fingerprint(ROOT/name)!=digest:raise ValueError('Products accepted sources changed')
        if fingerprint(__file__)!=identity['script_sha256'] or fingerprint(profile_path)!=identity['profile_sha256']:raise ValueError('Products frozen script/profile changed')
        verify_originals();check_cuda();seed_everything(0);torch.set_num_threads(4)
        model_path=Path(read_json_file(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint']);model_hash=fingerprint(model_path)
        if model_hash!=profile['encoder_sha256']:raise ValueError('Frozen Products encoder changed')
        storage=read_json_file(ROOT/'configs/storage.json');cache=Path(storage['embedding_cache_root'])/model_hash[:20]/'ogbn-products'
        needed=0 if (cache/'fused_embeddings.npy').exists() else profile['resources']['new_embedding_bytes']+2**20
        budget=check_budget(additional_cache_bytes=needed)
        saved=torch.load(str(model_path),map_location='cpu',weights_only=False,mmap=True);config=copy.deepcopy(saved['config']);config.update(name='products_full_single_seed',seed=0,storage=storage)
        config['evaluation']=dict(dataset='ogbn-products',task='node',split_id=0,seed=0,selection_split='official_validation',profile_sha256=identity['profile_sha256'],script_sha256=identity['script_sha256'],epochs=2000,patience=100,probe_batch_size=0,test_used_for_selection=False)
        data=load_prepared('ogbn-products')
        if data.graph.num_nodes()!=2449029 or len(data.splits['test'])!=2213091:raise ValueError('Full Products official nodes/test required')
        parent=args.resume_head or model_path
        experiment=Experiment(config,data.audit,parent);ledger.update(state='encoding',active_run=str(experiment.directory),embedding_cache=str(cache),budget=budget);ledger['attempts'].append({'run_directory':str(experiment.directory),'parent_sha256':fingerprint(parent)});atomic_write_json(path,ledger)
        model=make_model(config);model.load_state_dict(saved['model']);del saved
        embeddings=exact_inference(model,data,cache,model_hash,experiment,capacity['summary']['selected_batch_size'])
        del model;data.graph=None;gc.collect();torch.cuda.empty_cache()
        train,valid,test=node_inputs(data,embeddings,0);valid=CachedProbeInput(embeddings,valid.labels,indices=valid.indices);valid.cache_validation(400,experiment.logger)
        ledger['state']='validation_head_training';atomic_write_json(path,ledger)
        resume=args.resume_head
        if resume is None:
            for attempt in reversed(ledger['attempts'][:-1]):
                loc=Path(attempt['run_directory'])/'checkpoints/location.json'
                if loc.exists():
                    saved_path=Path(read_json_file(loc)['checkpoint_directory'])/'probe_last.pt'
                    if saved_path.exists():resume=saved_path;break
            if resume:raise ValueError('Existing head checkpoint requires explicit --resume-head to bind the new run parent SHA')
        summary=train_linear_probe(train,valid,None,47,2000,100,0,experiment,resume=resume,evaluate_test=False)
        summary.update(dataset='ogbn-products',official_train_nodes=len(train),official_validation_nodes=len(valid),official_test_nodes=len(test),encoder_sha256=model_hash,embedding_cache=str(cache),diagnostic_only=True,paper_reproduction_complete=False)
        write_json(experiment.directory/'summary.json',summary);experiment.status('completed_products_validation')
        validation_run=experiment.directory;best=experiment.checkpoint_directory/'probe_best.pt';experiment.close();experiment=None
        lock_data={'locked_before_test':True,'test_used_for_selection':False,'classifier_checkpoint':str(best),'classifier_sha256':fingerprint(best),'validation_summary_sha256':fingerprint(validation_run/'summary.json'),'selected_epoch':summary['selected_epoch'],'validation_accuracy':summary['validation_accuracy'],'cache_state_sha256':fingerprint(cache/'inference_state.json')}
        lock_path=validation_run/'selection_lock.json';atomic_write_json(lock_path,lock_data)
        config=copy.deepcopy(config);config['name']='products_full_confirmation';config['evaluation']['selection_lock_sha256']=fingerprint(lock_path)
        experiment=Experiment(config,data.audit,best);ledger.update(state='full_official_test',active_run=str(experiment.directory),validation_run=str(validation_run));atomic_write_json(path,ledger)
        head=torch.nn.Linear(1024,47).cuda();head.load_state_dict(torch.load(str(best),map_location='cpu',weights_only=False)['classifier'])
        accuracy,loss,pred=score_probe(head,valid,save_predictions=True)
        if accuracy!=summary['validation_accuracy']:raise ValueError('Saved Products validation head changed')
        accuracy,loss,pred=score_probe(head,test,save_predictions=True)
        if len(pred)!=2213091:raise ValueError('Full official Products test prediction count required')
        np.savez_compressed(experiment.directory/'test_predictions.npz',prediction=pred,labels=test.labels,node_ids=test.indices)
        if float(np.mean(pred==test.labels))!=accuracy:raise ValueError('Independent Products accuracy recomputation failed')
        summary.update(test_accuracy=accuracy,test_loss=loss,test_evaluations=1,validation_run_directory=str(validation_run),selection_lock_sha256=fingerprint(lock_path))
        write_json(experiment.directory/'summary.json',summary);experiment.metric({'step':1,'validation_accuracy':summary['validation_accuracy'],'test_accuracy':accuracy})
        experiment.logger.info('Products official full test accuracy=%.8f, queries=%d',accuracy,len(pred));experiment.status('completed_products_official_confirmation')
        ledger.update(state='completed_verified_official_single_seed',summary=summary,run_directory=str(experiment.directory),test_evaluations=1);atomic_write_json(path,ledger)
    except BaseException as error:
        ledger.update(state='paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));atomic_write_json(path,ledger)
        if experiment:experiment.status(ledger['state'],error=str(error))
        raise
    finally:
        if experiment:experiment.close()
        if lock:lock.close()
        supervisor.close()


if __name__=='__main__':main()
