"""GPU workers for the frozen remaining-main-task delivery; no test-based tuning."""
import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
OUT=ROOT/'artifacts/remaining_tasks_delivery'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def resources():return read(ROOT/'configs/storage.json')


def encoder():
    from came_repro.author_loader import fingerprint
    file=Path(read(ROOT/'artifacts/workflow_state/remaining_main_pilot_current.json')['encoder_checkpoint'])
    if fingerprint(file)!='1fd5a614766cf52fe15cc9e0a9f577fbfdc07cd70de7bd6cc7a02834a1cde91e':raise ValueError('Protected encoder changed')
    return file


def products_data(direct):
    from came_repro.prepared_data import load_prepared
    from came_repro.exact_csc_sampling import ExactCSCGraph
    from came_repro.compact_csc import load_graph
    data=load_prepared('ogbn-products',build_graph=False)
    directory=Path(resources()['embedding_cache_root'])/'structural_indexes/products_csc32_v1'
    from came_repro.author_loader import fingerprint
    profile=read(ROOT/'configs/remaining_tasks_delivery_20261018.json')
    if fingerprint(directory/'manifest.json')!=profile['capacity']['index_manifest_sha256']:raise ValueError('Certified Products topology changed')
    data.graph=ExactCSCGraph(directory) if direct else load_graph(directory).long().formats('csc')
    return data,directory


def score_products(model,data,ids,batch,direct):
    import numpy as np
    import torch
    import dgl
    from came_repro.exact_csc_sampling import ExactCSCSampler
    from came_repro.stream_inference import batch_full_neighbor_embeddings
    sampler=ExactCSCSampler(2) if direct else dgl.dataloading.MultiLayerFullNeighborSampler(2)
    rows=[]
    with torch.no_grad():
        for start in range(0,len(ids),batch):
            from came_repro.pause_control import pause_if_requested
            pause_if_requested()
            inputs,targets,blocks=sampler.sample_blocks(data.graph,torch.as_tensor(ids[start:start+batch],dtype=torch.int64))
            value=batch_full_neighbor_embeddings(model,data,blocks,inputs)
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite exact Products output')
            rows.append(value.cpu().numpy())
    return np.concatenate(rows)


def product_stage(stage):
    import numpy as np
    import torch
    import psutil
    from came_repro.train import make_model
    from came_repro.experiment import Experiment
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    saved=torch.load(str(encoder()),map_location='cpu',weights_only=False,mmap=True)
    config=copy.deepcopy(saved['config']);config.update(name='delivery_'+stage,seed=0,storage=resources())
    config['evaluation']=dict(dataset='ogbn-products',task='node',stage=stage,test_used_for_selection=False,
        no_test_labels=True,structural_adapter='exact_inbound_CSC_full_neighbors_local_DGL_blocks',precision='FP32',width=1024)
    config['deviations']=list(config.get('deviations',[]))+['Exact CSC neighbor enumeration avoids global DGL sampling allocations; all neighbors/parallel edges/loops retained, numerical acceptance required.']
    data,directory=products_data(stage!='products_reference');e=Experiment(config,data.audit,encoder())
    try:
        model=make_model(config);model.load_state_dict(saved['model']);model.eval();del saved
        cache=Path(resources()['embedding_cache_root'])/'1fd5a614766cf52fe15c/ogbn-products'
        if stage=='products_reference':
            ptr=np.load(directory/'indptr.npy',mmap_mode='r');top=np.argsort(np.diff(ptr))[-4:]
            state=read(cache/'inference_state.json');prefix=np.linspace(0,state['cursor']-1,16,dtype=np.int64)
            ids=np.unique(np.concatenate((top,np.random.default_rng(217).choice(2449029,12,replace=False),prefix)))
            value=score_products(model,data,ids,1,False)
            old=np.load(cache/'fused_embeddings.npy',mmap_mode='r')
            np.testing.assert_allclose(value[np.searchsorted(ids,prefix)],old[prefix],rtol=2e-5,atol=2e-5)
            file=OUT/'products_reference.npz';np.savez_compressed(file,node_ids=ids,embeddings=value)
            result=dict(reference_sha256=fingerprint(file),encoder_sha256=fingerprint(encoder()),
                index_manifest_sha256=fingerprint(directory/'manifest.json'),full_graph_nodes=2449029,width=1024,test_evaluations=0)
            atomic_write_json(OUT/'products_reference_identity.json',result)
        elif stage=='products_calibration':
            identity=read(OUT/'products_reference_identity.json');file=OUT/'products_reference.npz'
            prior=read(ROOT/'artifacts/workflow_state/remaining_tasks_delivery_current.json')['records']['products_reference']['summary']
            if identity!=prior:raise ValueError('Reference receipt no longer matches frozen reference run')
            if fingerprint(file)!=identity['reference_sha256'] or fingerprint(directory/'manifest.json')!=identity['index_manifest_sha256']:
                raise ValueError('Frozen exact Products reference changed')
            with np.load(file) as ref:ids=ref['node_ids'];expected=ref['embeddings']
            selected=None;measurements=[]
            for batch in [1,4,8,16,32,64]:
                try:
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    actual=score_products(model,data,ids,batch,True)
                    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
                    difference=float(np.max(np.abs(actual-expected)));selected=batch
                    measurements.append(dict(batch=batch,max_abs_difference=difference,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
                    e.metric(dict(step=batch,max_abs_difference=difference,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20))
                except torch.cuda.OutOfMemoryError:
                    gc.collect();torch.cuda.empty_cache();break
            if selected is None:raise RuntimeError('Exact one-node Products computation exceeds GPU capacity')
            cursor=read(cache/'inference_state.json')['cursor'];start_cursor=min(cursor,2449029-512);ids=np.arange(start_cursor,start_cursor+512)
            torch.cuda.synchronize();start=time.perf_counter();score_products(model,data,ids,selected,True);torch.cuda.synchronize()
            throughput=len(ids)/(time.perf_counter()-start);private=getattr(psutil.Process().memory_info(),'private',psutil.Process().memory_info().rss)
            result=dict(ready=throughput>=30 and private<.8*psutil.virtual_memory().total,
                nodes_per_second=throughput,private_bytes=private,selected_batch_size=selected,measurements=measurements,
                reference_identity=identity,full_graph_nodes=2449029,full_test_nodes=2213091,width=1024,test_evaluations=0)
            atomic_write_json(OUT/'products_capacity.json',result)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(9,3.8))
            axes[0].plot([r['batch'] for r in measurements],[r['peak_cuda_bytes']/2**30 for r in measurements],marker='o')
            axes[0].set(xlabel='Exact target batch size',ylabel='Peak CUDA memory (GiB)')
            axes[1].bar(['Measured','Required'],[throughput,30],color=['#315a7d','#b8b8b8']);axes[1].set(ylabel='Nodes / second')
            fig.text(.02,.01,'Products full topology /1024 FP32;512 contiguous nodes; no test scores; exact CSC adapter.',fontsize=8)
            fig.tight_layout(rect=(0,.08,1,1));fig.savefig(e.directory/'figures/products_capacity.png',dpi=180);plt.close(fig)
        else:
            receipt=read(OUT/'products_capacity.json')
            prior=read(ROOT/'artifacts/workflow_state/remaining_tasks_delivery_current.json')['records']['products_calibration']['summary']
            if receipt!=prior:raise ValueError('Capacity receipt no longer matches frozen calibration run')
            if not receipt['ready']:raise RuntimeError('Products capacity has not passed; full official data retained')
            from came_repro.exact_csc_sampling import ExactCSCSampler
            import dgl
            from unresolved_products_full_20261018 import exact_inference
            old=dgl.dataloading.MultiLayerFullNeighborSampler;dgl.dataloading.MultiLayerFullNeighborSampler=ExactCSCSampler
            try:exact_inference(model,data,cache,fingerprint(encoder()),e,receipt['selected_batch_size'])
            finally:dgl.dataloading.MultiLayerFullNeighborSampler=old
            result=dict(full_nodes=2449029,width=1024,inference_state_sha256=fingerprint(cache/'inference_state.json'),test_evaluations=0)
        atomic_write_json(e.directory/'summary.json',result);e.status('completed_'+stage)
    except BaseException as error:
        from came_repro.pause_control import TrainingPaused
        e.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
    finally:e.close()


def acceptance():
    import torch
    import numpy as np
    import dgl
    from came_repro.exact_csc_sampling import ExactCSCGraph
    from came_repro.compact_csc import build_index
    from came_repro.prepared_data import PreparedData
    from came_repro.train import make_model
    from came_repro.experiment import Experiment,seed_everything
    from came_repro.author_loader import fingerprint
    from came_repro.atomic_io import atomic_write_json
    saved=torch.load(str(encoder()),map_location='cpu',weights_only=False,mmap=True)
    config=copy.deepcopy(saved['config']);config.update(name='remaining_delivery_acceptance',seed=0,storage=resources())
    config['evaluation']=dict(fixture_only_not_paper_result=True,source='synthetic seven-node graph',release_manifest_sha256=fingerprint(ROOT/'artifacts/delivery/remaining_tasks_release.json'))
    e=Experiment(config,dict(fixture_only_not_paper_result=True),encoder())
    try:
        edges=np.array([[0,1],[0,1],[1,2],[2,0],[3,3],[3,3],[4,2]])
        directory=OUT/'fixture_csc';build_index(edges,7,directory,'delivery_fixture')
        features=np.random.default_rng(812).normal(size=(7,1024)).astype(np.float32)
        data=PreparedData(ExactCSCGraph(directory),features,None,{},{});model=make_model(config);model.load_state_dict(saved['model']);model.eval()
        with torch.no_grad():
            raw=dgl.graph((edges[:,0],edges[:,1]),num_nodes=7).to('cuda')
            expected=model(raw,torch.from_numpy(features[:,:512]).cuda(),torch.from_numpy(features[:,512:]).cuda(),getz3=True,compute_loss=False).cpu().numpy()
            actual=score_products(model,data,np.array([5,0,1,6,2,4,3]),3,True)[np.argsort([5,0,1,6,2,4,3])]
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        if not np.isfinite(actual).all():raise ValueError('Nonfinite exact sampler')
        initial=[]
        for seed in [0,1]:
            seed_everything(seed);head=torch.nn.Linear(2048,11).cuda();initial.append(head.weight.detach().cpu().clone())
        if torch.equal(*initial):raise ValueError('Distinct downstream seeds must create distinct initial weights')
        from came_repro.evaluation_cursor import EvaluationCursor
        from came_repro import link_evaluation as link
        from came_repro.remaining_lp_pilot import use_cached_vectors
        from came_repro.lp_validation_repair import make_head
        from came_repro.task_metrics import ranking_metrics
        seed_everything(731);vectors=torch.randn(17,1024,device='cuda');rank_head=make_head('two_hidden_layernorm_head_upstream_schedule').cuda();rank_head.eval()
        split=dict(source_node=np.array([0,1,2]),target_node=np.array([3,4,5]),
            target_node_neg=np.array([np.resize([3,6,7],150),np.resize([4,8,8],150),np.resize([5,9,10],150)]))
        with use_cached_vectors(),torch.no_grad():
            baseline=link.evaluate_split(rank_head,vectors,split,2,np.zeros(3,dtype=bool))
            def ranks(start,stop):
                a=link.vectors(vectors,split['source_node'][start:stop]);b=link.vectors(vectors,split['target_node'][start:stop])
                negatives=link.vectors(vectors,split['target_node_neg'][start:stop].ravel()).reshape(stop-start,150,-1)
                return (1/ranking_metrics(rank_head(a,b),rank_head(a[:,None,:].expand_as(negatives),negatives))['mrr']).cpu().numpy()
            one=e.checkpoint_directory/'fixture_eval_first';two=e.checkpoint_directory/'fixture_eval_resume';one.mkdir();two.mkdir()
            identity=dict(fixture_only=True,frozen_head='synthetic_fixed_head',canonical_scoring_block=2)
            cursor=EvaluationCursor(one/'test_cursor.json',identity,3,'float64');cursor.commit(0,ranks(0,2))
            resumed=EvaluationCursor(two/'test_cursor.json',identity,3,'float64',one/'test_cursor.json')
            if resumed.cursor!=2:raise ValueError('Official evaluation cursor did not restore')
            resumed.commit(2,ranks(2,3));rank_values=resumed.complete_values()
            if abs(float(np.mean(1./rank_values))-baseline['official']['mrr'])>1e-12:raise ValueError('Resumable official tie ranking differs')
        difference=float(np.max(np.abs(actual-expected)))
        s=dict(passed=True,actual_Torch_DGL_CUDA=True,full_author_model_and_direct_CSC_max_abs_difference=difference,
            distinct_seed_initialization_verified=True,resumable_GPU_ranking_and_official_ties_verified=True,
            fixture_not_benchmark=True,test_evaluations=0)
        e.metric(dict(step=1,max_abs_difference=difference));atomic_write_json(e.directory/'summary.json',s)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,3.4));ax.bar(np.arange(7),np.max(np.abs(actual-expected),axis=1),color='#315a7d')
        ax.set(xlabel='Synthetic node ID',ylabel='Maximum absolute FP32 difference',title='Full author model vs exact CSC blocks')
        fig.text(.02,.01,'7-node synthetic CUDA correctness fixture; parallel edges/loops/isolated nodes; not a dataset score.',fontsize=8)
        fig.tight_layout(rect=(0,.1,1,1));fig.savefig(e.directory/'figures/exact_cuda_fixture.png',dpi=180);plt.close(fig)
        e.status('completed_delivery_acceptance')
    except BaseException as error:e.status('failed',error=str(error));raise
    finally:e.close()


def train_task(task,seed,selection,resume):
    import torch
    import numpy as np
    from came_repro.experiment import Experiment
    from came_repro.atomic_io import atomic_write_json
    from came_repro.author_loader import fingerprint
    item=read(selection)['tasks'][task];reference=Path(item['validation_run']) if item.get('validation_run') else ROOT
    if task=='wn':
        from came_repro import kg_affine_probe as kg,experiment as exp
        original_kg,original_seed,original_solver=kg.Experiment,exp.seed_everything,kg.train_linear_probe
        def factory(config,audit,parent=None):
            config=copy.deepcopy(config);config['name']='delivery_WN_seed'+str(seed)
            config['evaluation'].update(delivery_selection_sha256=fingerprint(selection))
            return original_kg(config,audit,parent)
        kg.Experiment=factory
        if item['training_kind']=='normalized_wolfe':
            from remaining_objective_backend_20261018 import stable_linear
            kg.train_linear_probe=stable_linear
            exp.seed_everything=lambda ignored:original_seed(seed)
        previous=sys.argv;sys.argv=['kg_affine_probe','--dataset','WN18RR','--checkpoint',str(encoder()),'--embedding-cache',item['embedding_cache'],'--graph-policy','train_reverse','--seed',str(seed)]
        if resume:sys.argv+=['--resume-head',str(resume)]
        created=[]
        def tracked_factory(config,audit,parent=None):
            result=factory(config,audit,parent);created.append(result.directory);return result
        kg.Experiment=tracked_factory
        try:
            kg.main()
            if item['training_kind']=='normalized_wolfe':
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                run=created[-1];rows=[json.loads(line) for line in (run/'logs/metrics.jsonl').read_text().splitlines()]
                summary=read(run/'summary.json');fig,axes=plt.subplots(1,2,figsize=(10,4))
                for key,label in [('train_loss','train'),('val_loss','validation')]:axes[0].plot([r['step'] for r in rows],[r[key] for r in rows],label=label)
                axes[0].legend();axes[0].set(xlabel='Optimizer update',ylabel='Cross entropy')
                axes[1].plot([r['step'] for r in rows],[r['val_accuracy'] for r in rows]);axes[1].axvline(summary['selected_epoch'],ls='--',color='black')
                axes[1].set(xlabel='Optimizer update',ylabel='Official validation accuracy')
                fig.text(.02,.01,'WN18RR; seed%d; fixed unit-L2 embeddings; linear CE; closure counts logged; test held out.'%seed,fontsize=8)
                fig.tight_layout(rect=(0,.08,1,1));fig.savefig(run/'figures/probe_curves.png',dpi=180);plt.close(fig)
        finally:sys.argv=previous;kg.Experiment=original_kg;exp.seed_everything=original_seed;kg.train_linear_probe=original_solver
        return
    if task=='products':
        from came_repro.prepared_data import load_prepared
        from came_repro.downstream import node_inputs,train_linear_probe
        from came_repro.quality_probe import CachedProbeInput
        data=load_prepared('ogbn-products',build_graph=False)
        cache=Path(resources()['embedding_cache_root'])/'1fd5a614766cf52fe15c/ogbn-products'
        state=read(cache/'inference_state.json')
        if not state['complete'] or fingerprint(cache/'fused_embeddings.npy')!=state['output_sha256']:raise ValueError('Verified complete Products matrix required')
        matrix=np.load(cache/'fused_embeddings.npy',mmap_mode='r')
        config=copy.deepcopy(torch.load(str(encoder()),map_location='cpu',weights_only=False,mmap=True)['config'])
        config.update(name='delivery_Products_seed'+str(seed),seed=seed,storage=resources())
        config['evaluation']=dict(dataset='ogbn-products',task='node',seed=seed,selection_split='official_validation',delivery_selection_sha256=fingerprint(selection))
        e=Experiment(config,data.audit,resume or encoder())
        try:
            train,valid,_=node_inputs(data,matrix,0)
            valid=CachedProbeInput(matrix,valid.labels,indices=valid.indices);valid.cache_validation(400,e.logger)
            s=train_linear_probe(train,valid,None,47,2000,100,seed,e,resume=resume,evaluate_test=False)
            s.update(dataset='ogbn-products',encoder_sha256=fingerprint(encoder()),embedding_cache=str(cache))
            atomic_write_json(e.directory/'summary.json',s);e.status('completed_products_validation')
        except BaseException as error:
            from came_repro.pause_control import TrainingPaused
            e.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
        finally:e.close()
        return
    from came_repro.lp_validation_repair import load_verified_cache
    from came_repro.remaining_lp_pilot import use_cached_vectors
    dataset=item['dataset'];cache=Path(item['embedding_cache']);gpu,audit=load_verified_cache(dataset,cache,fingerprint(encoder()))
    raw=ROOT/'data/raw/mm_graph'/dataset/'lp-edge-split.pt'
    if fingerprint(raw)!=audit['raw_sha256']['lp-edge-split.pt']:raise ValueError('Official LP split changed')
    split=torch.load(str(raw),map_location='cpu',weights_only=False,mmap=True)
    config=read(reference/'config.json');config.update(name='delivery_'+task+'_seed'+str(seed),seed=seed)
    recipe=item['recipe'];config['evaluation'].update(recipe,seed=seed,delivery_selection_sha256=fingerprint(selection))
    e=Experiment(config,audit,resume or encoder())
    try:
        if item['training_kind']=='listwise':from came_repro.listwise_link_train import train_listwise as train
        elif item['training_kind']=='dropout_bce':from came_repro.cloth_regularized_train import train_regularized as train
        elif item['training_kind']=='hard_four':from run_lp_hard_negative_candidate import train_hard as train
        else:from run_locked_lp_repeats import train_repeat as train
        with use_cached_vectors():s=train(gpu,split['train'],split['valid'],recipe,seed,e,resume)
        s.update(dataset=dataset,task='link',encoder_sha256=fingerprint(encoder()),embedding_cache=str(cache),graph_policy=item['graph_policy'])
        atomic_write_json(e.directory/'summary.json',s);e.status('completed_delivery_validation')
    except BaseException as error:
        from came_repro.pause_control import TrainingPaused
        e.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
    finally:e.close()




def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['acceptance','products_reference','products_calibration','products_extract','train','test'],required=True)
    parser.add_argument('--task',choices=['wn','cloth','books','products']);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--selection',type=Path);parser.add_argument('--resume-head',type=Path);args=parser.parse_args()
    from came_repro.protocol_repair import check_cuda
    from came_repro.author_loader import verify_originals
    from remaining_delivery import verify_release
    verify_release();verify_originals();check_cuda()
    import torch
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='acceptance':acceptance()
    elif args.stage.startswith('products_'):product_stage(args.stage)
    elif args.stage=='test':
        from remaining_delivery_test import confirm
        confirm(args.task,args.selection,args.resume_head)
    else:train_task(args.task,args.seed,args.selection,args.resume_head)


if __name__=='__main__':main()
