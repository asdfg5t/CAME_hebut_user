"""Read-only CUDA capacity/equivalence calibration for exact NC inference batches."""
import sys
from pathlib import Path
if __name__=='__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import argparse
import time
import gc
import numpy as np
import torch
import dgl
from came_repro.author_loader import ROOT, fingerprint, verify_originals
from came_repro.ide_workflow import acquire_gpu_lock
from came_repro.quality_pipeline import reference_assets
from came_repro.prepared_data import load_prepared
from came_repro.train import make_model
from came_repro.stream_inference import batch_full_neighbor_embeddings
from came_repro.atomic_io import atomic_write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,choices=[32,64,128],default=32)
    batch_size=parser.parse_args().batch_size
    lock=acquire_gpu_lock()
    try:
        verify_originals()
        model_path,_,_,_=reference_assets()
        state=torch.load(model_path,map_location='cpu',weights_only=False)
        model=make_model(state['config']).eval()
        model.load_state_dict(state['model']);del state;gc.collect()
        for parameter in model.parameters():parameter.requires_grad_(False)
        data=load_prepared('books-nc')
        data.graph=dgl.to_bidirected(data.graph)
        graph=model._prep_graph(data.graph)
        sampler=dgl.dataloading.MultiLayerFullNeighborSampler(2)
        degrees=graph.in_degrees()
        top=torch.topk(degrees,1000).indices.numpy()
        rng=np.random.default_rng(123)
        batches=[top[:batch_size],rng.choice(top,batch_size,replace=False),np.arange(52064,52064+batch_size),rng.choice(graph.num_nodes(),batch_size,replace=False)]
        results=[]
        with torch.no_grad():
            for index,targets in enumerate(batches):
                targets=torch.as_tensor(targets,dtype=torch.long)
                inputs,output,blocks=sampler.sample_blocks(graph,targets)
                if not torch.equal(output,targets):raise ValueError('Sampler target order changed')
                gc.collect()
                torch.cuda.empty_cache()
                free,_=torch.cuda.mem_get_info()
                forecast=len(inputs)*14*1024+512*2**20
                if forecast>free:
                    raise RuntimeError(f'Conservative batch{batch_size} memory forecast exceeds free CUDA budget')
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                actual=batch_full_neighbor_embeddings(model,data,blocks,inputs).cpu()
                torch.cuda.synchronize();elapsed=time.perf_counter()-started
                peak=torch.cuda.max_memory_allocated()/2**20
                expected=[];torch.cuda.synchronize();started=time.perf_counter()
                for start in range(0,batch_size,8):
                    a,b,c=sampler.sample_blocks(graph,targets[start:start+8])
                    expected.append(batch_full_neighbor_embeddings(model,data,c,a).cpu())
                expected=torch.cat(expected);torch.cuda.synchronize();old_seconds=time.perf_counter()-started
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                if peak>4800:raise RuntimeError('Batch32 leaves insufficient VRAM reserve')
                results.append({'sample':index,'input_nodes':len(inputs),'candidate_batch_compute_seconds':elapsed,
                                'batch8_sample_compute_seconds_for_same_targets':old_seconds,'peak_allocated_mib':peak,
                                'max_absolute_difference':float((actual-expected).abs().max())})
                print(json.dumps(results[-1]),flush=True)
        report={'dataset':'books-nc','graph_policy':'bidirected','encoder_sha256':fingerprint(model_path),
                'candidate_batch_size':batch_size,'previous_batch_size':8,'actual_cuda_checked':True,
                'output_equivalence_passed':True,'memory_calibration_passed':True,'samples':results,
                'scope':'degree-stress and distributed unlabeled nodes; no labels or test scores selected the batch size'}
        atomic_write_json(ROOT/f'artifacts/audit/books_inference_batch_calibration_{batch_size}.json',report)
    finally:
        lock.close()


if __name__=='__main__':
    main()
