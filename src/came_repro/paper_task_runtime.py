"""Owned runtime adapters; old source files and all active workflows remain untouched."""
import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from .author_loader import ROOT,fingerprint
from .atomic_io import atomic_write_json,unique_temporary_path,atomic_replace,retry_windows_io
from .experiment import Experiment
from .paper_task_groups import PROFILE,RELEASE,OUT,read,verify_release
from .paper_task_io import BufferedTensorBoardWriter,verified_validation_replay

CURRENT=None
LAST_CACHE=None
WRITER_BASELINES={}


def owned_cache_root():
    storage=read(ROOT/'configs/storage.json')
    return Path(storage['embedding_cache_root'])/'remaining_paper_tasks'/fingerprint(RELEASE)[:16]


def select_clip_view(features,representation):
    if representation not in ['clip_text','clip_image']:raise ValueError('Explicit CLIP modality required')
    if len(features.shape)!=2 or features.shape[1] not in [512,1024]:raise ValueError('Verified512/1024 feature layout required')
    if representation=='clip_image' and features.shape[1]!=1024:raise ValueError('No image baseline exists for a text-only graph')
    values=features[:,:512] if representation=='clip_text' else features[:,512:1024]
    return values.numpy() if isinstance(values,torch.Tensor) else values


def fast_checkpoint(path,state):
    from .storage_budget import check_budget,tensor_storage_bytes
    from .filesystem_paths import windows_io_path
    storage=read(ROOT/'configs/storage.json');root=Path(storage['large_checkpoint_root']).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root) or len(path.relative_to(root).parts)!=2 or '_paper_tasks__' not in path.parent.name:
        raise ValueError('Remaining-task checkpoint escaped owned run family')
    estimate=tensor_storage_bytes(state)+2**20
    if path.parent not in WRITER_BASELINES:
        report=check_budget(additional_cache_bytes=estimate)
        used=sum(x.stat().st_size for x in path.parent.iterdir() if x.is_file())
        WRITER_BASELINES[path.parent]=(report['external_cache_bytes']-used,used)
    baseline,initial_owned=WRITER_BASELINES[path.parent]
    def guard():
        used=sum(x.stat().st_size for x in path.parent.iterdir() if x.is_file())
        if (baseline+used+estimate>storage['external_cache_high_water_gib']*2**30
                or used+estimate-initial_owned>read(PROFILE)['own_checkpoint_reservation_gib']*2**30
                or shutil.disk_usage(root).free-estimate<storage['reserve_free_gib']*2**30):
            raise RuntimeError('Owned checkpoint reservation exhausted; no historical results removed')
    guard();temporary=unique_temporary_path(path)
    def save():
        with open(windows_io_path(temporary),'wb') as f:
            torch.save(state,f);f.flush();os.fsync(f.fileno())
    retry_windows_io(save,temporary);atomic_replace(temporary,path)


def install_adapters(job):
    """Per-child process only. In-place config metadata preserves resume identity."""
    from . import experiment,downstream,subset_offline_train,stream_inference
    original_init=experiment.Experiment.__init__
    original_extract=stream_inference.extract_full_embeddings
    original_score=downstream.score_probe
    original_load=downstream.load_prepared
    def init(self,config,audit,parent=None):
        global CURRENT
        config['name']='paper_tasks_'+config['name']
        config['paper_task_runtime']=dict(job_id=job['id'],release_sha256=fingerprint(RELEASE),
            validation_cache_mib=400,checkpoint_writer='owned_atomic_with_live_guard',
            tensorboard_flush_seconds=5,jsonl_csv_flush='every metric row',
            owned_cache_scope=str(owned_cache_root()),paper_parameters_and_reconstruction_details_separated=True)
        if job.get('encoder_checkpoint') and 'complete_pretraining' in config.get('evaluation',{}):
            parent_state=torch.load(job['encoder_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
            rounds=parent_state.get('completed_subset_rounds',0)
            if rounds!=8:raise ValueError('Eight complete declared subset rounds required')
            config['evaluation'].update(complete_pretraining=True,completed_pretraining_subset_rounds=rounds,
                completed_full_graph_epochs=0,pretraining_scope='Declared subset rounds; not full-graph epochs')
        original_init(self,config,audit,parent)
        self.writer=BufferedTensorBoardWriter(self.writer)
        CURRENT=self
        if LAST_CACHE and config.get('evaluation',{}).get('dataset')==LAST_CACHE['dataset']:
            atomic_write_json(self.directory/'embedding_source.json',LAST_CACHE)
    def load(*args,**kwargs):
        data=original_load(*args,**kwargs)
        if job.get('representation') in ['clip_text','clip_image'] and data.features.shape[1]==1024 and not isinstance(data.features,torch.Tensor):
            values=np.load(data.features.filename,mmap_mode='c') if hasattr(data.features,'filename') else np.array(data.features,copy=True)
            data.features=torch.from_numpy(values)
        return data
    def extract(model,data,directory,checkpoint_sha256,batch_size=64,log=None):
        global LAST_CACHE
        directory=Path(directory);state_path=directory/'inference_state.json'
        expected=dict(checkpoint_sha256=checkpoint_sha256,data=data.audit,hidden_dim=1024,
            method='All inbound neighbors, all encoder layers, eval mode; no PPR approximation',precision='fp32')
        compatible=state_path.exists() and read(state_path).get('complete') and read(state_path)['identity']==expected
        actual=directory if compatible else owned_cache_root()/checkpoint_sha256[:20]/data.audit['dataset']
        LAST_CACHE=dict(requested_hint=str(directory),actual_cache=str(actual),dataset=data.audit['dataset'],
            protected_existing_cache=bool(compatible),owned=not compatible,checkpoint_sha256=checkpoint_sha256)
        if CURRENT:atomic_write_json(CURRENT.directory/'embedding_source.json',LAST_CACHE)
        return original_extract(model,data,actual,checkpoint_sha256,batch_size,log)
    def score(classifier,data,batch_size=4096,save_predictions=False):
        required=len(data)*(data.width*4+8)
        if not hasattr(data,'_paper_cuda_cache') and required<=400*2**20 and required+512*2**20<torch.cuda.mem_get_info()[0]:
            features,labels=data.batch(np.arange(len(data)))
            data._paper_cuda_cache=(features,labels)
            def cached_batch(index):
                ids=torch.as_tensor(index,dtype=torch.long,device='cuda')
                return features[ids],labels[ids]
            data.batch=cached_batch
            if CURRENT:CURRENT.logger.info('Complete fixed validation/test input cached on CUDA: %.2fMiB; unchanged4096-row scoring',required/2**20)
        return original_score(classifier,data,batch_size,save_predictions)
    experiment.Experiment.__init__=init
    downstream.atomic_checkpoint=fast_checkpoint
    downstream.extract_full_embeddings=extract
    downstream.score_probe=score
    downstream.load_prepared=load
    subset_offline_train.atomic_checkpoint=fast_checkpoint
    return extract


def lp_plot(history,selected,e):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['step'] for r in history],[r['train_loss'] for r in history])
    axes[1].plot([r['step'] for r in history],[100*r['validation_mrr'] for r in history])
    axes[1].axvline(selected,color='black',ls='--')
    axes[0].set(xlabel='Declared sampled optimization round',ylabel='BCE loss')
    axes[1].set(xlabel='Declared sampled optimization round',ylabel='Complete official validation MRR (%)')
    fig.suptitle(e.config['evaluation']['dataset']+'; matched reconstructed LP protocol; test held out')
    fig.tight_layout();fig.savefig(e.directory/'figures/lp_training.png',dpi=180);plt.close(fig)


def lp_job(job,resume,extract):
    from .prepared_data import load_prepared
    from .train import make_model
    from . import goodreads_official as core
    from .evaluation_cursor import EvaluationCursor
    from .task_metrics import reverse_overlap_mask
    from .pause_control import TrainingPaused
    core.atomic_checkpoint=fast_checkpoint;core.draw_curves=lp_plot
    data=load_prepared(job['dataset'],build_graph=job['representation']=='came')
    storage=read(ROOT/'configs/storage.json')
    config=dict(name='matched_lp_'+job['representation']+'_'+job['dataset'],seed=job['seed'],storage=storage,
        evaluation=dict(dataset=job['dataset'],recipe=job['lp_recipe'],representation=job['representation'],seed=job['seed'],test_used_for_selection=False),
        deviations=['Shared public-MM-Graph LP head/optimizer recipe reconstructed; exact author LP trainer unreleased.',
            'Three-linear Hadamard/ReLU head; per-round512000-positive ceiling; these are not full official graph epochs.'])
    if job['representation']=='came':
        saved=torch.load(job['encoder_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        if saved.get('completed_subset_rounds')!=8:raise ValueError('Incomplete ablation encoder cannot unlock downstream test')
        config['encoder_checkpoint_sha256']=fingerprint(job['encoder_checkpoint'])
        model=make_model(saved['config']);model.load_state_dict(saved['model']);del saved
        hint=Path(storage['embedding_cache_root'])/config['encoder_checkpoint_sha256'][:20]/job['dataset']
        embeddings=extract(model,data,hint,config['encoder_checkpoint_sha256'],32)
        del model;data.graph=None;gc.collect();torch.cuda.empty_cache()
    else:embeddings=select_clip_view(data.features,job['representation'])
    if embeddings.nbytes+int(1.5*2**30)>torch.cuda.mem_get_info()[0]:raise RuntimeError('Full FP32 LP representation exceeds CUDA reserve; no CPU fallback')
    gpu=torch.empty(embeddings.shape,dtype=torch.float32,device='cuda')
    for start in range(0,len(embeddings),8192):
        gpu[start:start+8192]=torch.from_numpy(np.array(embeddings[start:start+8192],dtype=np.float32,copy=True)).cuda()
    core.require_finite_rows(gpu)
    raw=ROOT/'data/raw/mm_graph'/job['dataset']/'lp-edge-split.pt'
    if fingerprint(raw)!=data.audit['raw_sha256']['lp-edge-split.pt']:raise ValueError('Official LP split SHA changed')
    split=torch.load(str(raw),map_location='cpu',weights_only=False,mmap=True)
    for key in ['valid','test']:
        if np.asarray(split[key]['target_node_neg']).shape!=(len(split[key]['source_node']),150):raise ValueError('Full fixed150 official candidates required')
    cursor_parent=None
    if resume and Path(resume).suffix=='.json':
        old=read(resume)
        if 'identity' in old:
            cursor_parent=resume;lock_path=Path(old['identity']['selection_lock_path'])
        else:lock_path=Path(resume)
        lock=read(lock_path)
    else:
        e=Experiment(copy.deepcopy(config),data.audit,resume or Path(job['encoder_checkpoint']) if job.get('encoder_checkpoint') else resume)
        try:
            summary=core.train_rounds(gpu,split['train'],split['valid'],job['lp_recipe'],job['seed'],e,resume)
            summary.update(dataset=job['dataset'],representation=job['representation'],matched_protocol=True,
                head_variant='pinned_sage_predictor_'+str(gpu.shape[1]),paper_reproduction_complete=False)
            atomic_write_json(e.directory/'validation_summary.json',summary)
            head=e.checkpoint_directory/'link_head_best.pt'
            lock=dict(classifier_checkpoint=str(head),classifier_sha256=fingerprint(head),validation_run_directory=str(e.directory),
                validation_summary_sha256=fingerprint(e.directory/'validation_summary.json'),
                validation_ranks_sha256=fingerprint(e.directory/'validation_link_ranks.npz'),
                data_audit=copy.deepcopy(data.audit),selected_head_validation_replayed=True,
                locked_before_test=True,test_used_for_selection=False)
            # A completed validation phase is not a completed downstream task.
            e.status('completed_matched_lp_validation')
            lock_path=e.directory/'selection_lock.json';atomic_write_json(lock_path,lock)
        except BaseException as error:e.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
        finally:e.close()
    if not lock['locked_before_test'] or lock['test_used_for_selection'] or fingerprint(lock['classifier_checkpoint'])!=lock['classifier_sha256']:
        raise ValueError('Frozen validation-selected LP head required')
    validation=Path(lock['validation_run_directory'])
    if fingerprint(validation/'validation_summary.json')!=lock['validation_summary_sha256']:raise ValueError('Validation summary changed')
    summary,selected_head=verified_validation_replay(lock,split['valid'],data.audit)
    test_config=copy.deepcopy(config);test_config['name']='matched_lp_test_'+job['dataset'];test_config['evaluation']['phase']='test'
    e=Experiment(test_config,data.audit,cursor_parent or Path(lock['classifier_checkpoint']))
    try:
        atomic_write_json(e.directory/'selection_lock.json',lock)
        head=core.make_head(gpu.shape[1]).cuda()
        head.load_state_dict(selected_head['head'])
        e.logger.info('Complete selected-head validation replay reused after head/data/rank SHA and full official IDs/MRR verification; no duplicate GPU pass')
        identity=dict(selection_lock_path=str(lock_path),classifier_sha256=lock['classifier_sha256'],dataset=job['dataset'],seed=job['seed'],canonical_block=64)
        if cursor_parent is None:
            from .storage_budget import guard_allocation
            guard_allocation(e.checkpoint_directory/'test_values.npy',len(split['test']['source_node'])*8+2**20)
        cursor=EvaluationCursor(e.checkpoint_directory/'test_cursor.json',identity,len(split['test']['source_node']),'float64',cursor_parent)
        ranks=core.evaluate(head,gpu,split['test'],e,cursor)
        mask=reverse_overlap_mask(np.asarray(split['train']['source_node']),np.asarray(split['train']['target_node']),
            np.asarray(split['test']['source_node']),np.asarray(split['test']['target_node']),len(gpu))
        np.savez_compressed(e.directory/'test_link_ranks.npz',source_node=split['test']['source_node'],target_node=split['test']['target_node'],rank=ranks,reverse_overlap=mask)
        summary.update(test=dict(official=core.metrics(ranks),queries=len(ranks),reverse_overlap_queries=int(mask.sum()),reverse_nonoverlap_subset=core.metrics(ranks[~mask])),
            test_evaluations=1,selection_lock_sha256=fingerprint(lock_path),validation_run_directory=str(validation),
            sampler_and_optimizer_are_declared_reconstruction=True)
        summary['test_preflight_validation']='Reused immutable full selected-head replay; head/data/IDs/ranks/MRR independently verified'
        shutil.copyfile(validation/'figures/lp_training.png',e.directory/'figures/lp_training.png')
        atomic_write_json(e.directory/'summary.json',summary)
        e.metric(dict(step=len(ranks),test_mrr=summary['test']['official']['mrr'],official_queries=len(ranks),test_evaluations=1))
        e.logger.info('COMPLETE_MATCHED_LP dataset=%s view=%s seed=%d MRR=%.8f%% full_queries=%d',job['dataset'],job['representation'],job['seed'],100*summary['test']['official']['mrr'],len(ranks))
        e.status('completed_paper_lp_test')
    except BaseException as error:e.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
    finally:e.close()


def main():
    from .protocol_repair import check_cuda
    from .paper_task_groups import jobs_for_group,GROUPS
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);parser.add_argument('--resume',type=Path)
    args=parser.parse_args();verify_release();job=read(args.job)
    expected=next((j for group in GROUPS for j in jobs_for_group(group) if j['id']==job['id']),None)
    stripped={k:v for k,v in job.items() if k!='encoder_checkpoint'}
    if expected!=stripped:raise ValueError('Resolved scientific job differs from frozen paper grid')
    check_cuda();torch.set_num_threads(4)
    extract=install_adapters(job)
    if job['kind']=='pretraining':
        from . import subset_offline_train
        path=args.job.with_name(args.job.stem+'_train_config.json')
        if path.exists() and read(path)!=job['config']:raise ValueError('Resolved pretraining config changed')
        atomic_write_json(path,job['config'])
        sys.argv=[sys.argv[0],'--config',str(path)]
        if args.resume:sys.argv+=['--resume',str(args.resume)]
        subset_offline_train.main()
    elif job.get('task')=='link':lp_job(job,args.resume,extract)
    else:
        from . import downstream
        sys.argv=[sys.argv[0],'--dataset',job['dataset'],'--representation',job['representation'],'--task',job['task'],
            '--seed',str(job['seed']),'--split-id',str(job['split_id']),'--epochs','2000','--patience','100','--inference-batch-size','32']
        if job.get('encoder_checkpoint'):sys.argv+=['--checkpoint',job['encoder_checkpoint']]
        if args.resume:sys.argv+=['--resume-head',str(args.resume)]
        downstream.main()
    if CURRENT and (CURRENT.directory/'summary.json').exists():
        summary=read(CURRENT.directory/'summary.json')
        source=CURRENT.directory/'embedding_source.json'
        if source.exists():
            summary['actual_embedding_cache']=read(source)['actual_cache']
        if job.get('encoder_checkpoint'):
            saved=torch.load(job['encoder_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
            summary.update(completed_pretraining_subset_rounds=saved.get('completed_subset_rounds',0),
                pretraining_scope='declared fixed subset rounds; not full-graph epochs')
        atomic_write_json(CURRENT.directory/'summary.json',summary)


if __name__=='__main__':main()
