"""Measure the unchanged paper model on a verified large MM graph before long runs."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from came_repro.prepared_data import load_prepared
from came_repro.joint_sampling import PackedPPRSampler
from came_repro.train import make_model
from came_repro.experiment import Experiment,seed_everything,write_json,atomic_checkpoint,rng_state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['books-lp','books-nc','cloth-copurchase'],default='books-lp')
    parser.add_argument('--seed-batch-size',type=int,default=32)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    config=json.loads((ROOT/'configs/joint_representation_reconstruction.json').read_text())
    config['name']='joint_resource_'+args.dataset.replace('-','_')
    config['stage']='verified_corpus_resource_probe'
    config['storage']=json.loads((ROOT/'configs/storage.json').read_text())
    config['sampling']['seed_batch_size']=args.seed_batch_size
    seed_everything(config['seed'])
    data=load_prepared(args.dataset,require_verified=True)
    experiment=Experiment(config,data.audit)
    sampler=PackedPPRSampler(data,config['sampling'])
    try:
        seeds=torch.randperm(data.graph.num_nodes(),generator=torch.Generator().manual_seed(config['seed']))[:args.seed_batch_size]
        started=time.perf_counter()
        graph,nodes,residual=sampler.sample(seeds.tolist())
        sampling=time.perf_counter()-started
        model=make_model(config)
        optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
        a,b=data.modality_batch(nodes)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        loss,outputs=model(graph.to('cuda'),a.cuda(),b.cuda(),return_embeddings=True)
        loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        optimizer.step()
        torch.cuda.synchronize()
        result={'step':1,'dataset':args.dataset,'seed_count':len(seeds),'subgraph_nodes':len(nodes),'subgraph_edges':graph.num_edges(),
                'loss':float(loss),'loss_tv':float(outputs['loss12']),'loss_tf':float(outputs['loss13']),'loss_vf':float(outputs['loss23']),
                'grad_norm':norm,'lr':.0001,'sampling_seconds':sampling,'step_seconds':time.perf_counter()-started,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                'ppr_residual_max':residual}
        experiment.metric(result)
        write_json(experiment.directory/'summary.json',{'resource_probe':result,'paper_reproduction_complete':False,'full_pretraining_started':False})
        # Resource probes need metrics, not another 600 MiB optimizer checkpoint.
        experiment.status('completed_resource_probe')
        print(json.dumps(result),flush=True)
        print(f'RUN_DIR={experiment.directory}',flush=True)
    except BaseException as error:
        experiment.logger.exception('Resource probe failed')
        experiment.status('failed',error=str(error))
        raise
    finally:
        sampler.close()
        experiment.close()


if __name__=='__main__':
    main()
