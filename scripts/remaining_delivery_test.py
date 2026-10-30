"""Exactly once per committed official query, with a durable deterministic cursor."""
from pathlib import Path
import copy
import json
import time


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def confirm(task,lock_file,resume=None):
    import numpy as np
    import torch
    from came_repro.author_loader import ROOT,fingerprint
    from came_repro.atomic_io import atomic_write_json
    from came_repro.experiment import Experiment
    from came_repro.evaluation_cursor import EvaluationCursor
    from came_repro.prepared_data import prepared_directory,load_prepared
    from came_repro.lp_validation_repair import load_verified_cache,make_head
    from came_repro.remaining_lp_pilot import use_cached_vectors
    from came_repro.task_metrics import ranking_metrics,reverse_overlap_mask
    from came_repro.pause_control import pause_if_requested,TrainingPaused
    from came_repro import link_evaluation as link
    from came_repro.downstream import score_probe,node_inputs
    lock=read(lock_file);selected=Path(lock['validation_run_directory']);summary=read(selected/'summary.json')
    if (not lock['locked_before_test'] or lock['test_used_for_selection'] or summary['test_evaluations']!=0
            or fingerprint(lock['classifier_checkpoint'])!=lock['classifier_sha256']
            or fingerprint(selected/'summary.json')!=lock['validation_summary_sha256']):raise ValueError('Validation-selected test lock changed')
    cache=Path(lock['embedding_cache']);state=read(cache/'inference_state.json');data=copy.deepcopy(state['identity']['data'])
    if (not state['complete'] or state['identity']['checkpoint_sha256']!=lock['encoder_sha256']
            or fingerprint(cache/'inference_state.json')!=lock['inference_state_sha256']):raise ValueError('Locked embedding identity changed')
    config=read(selected/'config.json');config['name']='delivery_'+task+'_test_s'+str(summary['seed'])
    config['evaluation'].update(selection_lock_sha256=fingerprint(lock_file),test_resume='fixed head/data; committed output prefix never rescored')
    e=Experiment(config,data,Path(resume) if resume else Path(lock['classifier_checkpoint']))
    try:
        if task in ['cloth','books']:
            dataset=lock['dataset'];gpu,_=load_verified_cache(dataset,cache,lock['encoder_sha256'])
            raw=ROOT/'data/raw/mm_graph'/dataset/'lp-edge-split.pt'
            if fingerprint(raw)!=data['raw_sha256']['lp-edge-split.pt']:raise ValueError('Official LP split changed')
            split=torch.load(str(raw),map_location='cpu',weights_only=False,mmap=True)
            test=split['test'];source,target,negative=(np.asarray(test[k]) for k in ['source_node','target_node','target_node_neg'])
            if negative.shape!=(len(source),150):raise ValueError('Complete fixed150 negatives required')
            overlap=reverse_overlap_mask(np.asarray(split['train']['source_node']),np.asarray(split['train']['target_node']),source,target,len(gpu))
            head=make_head(lock['head_variant']).cuda();head.load_state_dict(torch.load(lock['classifier_checkpoint'],map_location='cpu',weights_only=False)['head']);head.eval()
            with use_cached_vectors(),torch.no_grad():
                valid=link.evaluate_split(head,gpu,split['valid'],64,np.zeros(len(split['valid']['source_node']),dtype=bool))
            if abs(valid['official']['mrr']-summary['validation_mrr'])>1e-12:raise ValueError('Frozen LP validation replay changed')
            n=len(source);batch=64;dtype='float64'
        else:
            matrix=np.load(cache/'fused_embeddings.npy',mmap_mode='r')
            if fingerprint(cache/'fused_embeddings.npy')!=state['output_sha256']:raise ValueError('Full frozen output checksum changed')
            if task=='wn':
                from came_repro.kg_graph_probe import kg_input
                from came_repro.kg_selected_confirmation import RawDoubleInput
                directory=prepared_directory('WN18RR');manifest=read(directory/'manifest.json')
                if fingerprint(directory/'manifest.json')!=data['prepared_manifest_sha256']:raise ValueError('Official KG manifest changed')
                for name,digest in manifest['files'].items():
                    if fingerprint(directory/name)!=digest:raise ValueError('Official KG asset changed')
                raw_valid=kg_input('WN18RR',matrix,'valid');raw_test=kg_input('WN18RR',matrix,'test')
                valid=RawDoubleInput(matrix,raw_valid.labels,pairs=raw_valid.pairs);test=RawDoubleInput(matrix,raw_test.labels,pairs=raw_test.pairs)
                head=torch.nn.Linear(2048,manifest['relations'],dtype=torch.float64).cuda();classes=manifest['relations']
                if len(test)!=3134:raise ValueError('Complete official WN test required')
            else:
                prepared=load_prepared('ogbn-products',build_graph=False);_,valid,test=node_inputs(prepared,matrix,0)
                head=torch.nn.Linear(1024,47).cuda();classes=47
                if len(test)!=2213091:raise ValueError('Complete official Products test required')
            head.load_state_dict(torch.load(lock['classifier_checkpoint'],map_location='cpu',weights_only=False)['classifier']);head.eval()
            value,_,_=score_probe(head,valid)
            if value!=summary['validation_accuracy']:raise ValueError('Frozen classification validation replay changed')
            n=len(test);batch=4096;dtype='int64'
        identity=dict(selection_lock_sha256=fingerprint(lock_file),head_sha256=lock['classifier_sha256'],
            cache_state_sha256=lock['inference_state_sha256'],task=task,canonical_scoring_block=batch)
        cursor=EvaluationCursor(e.checkpoint_directory/'test_cursor.json',identity,n,dtype,resume)
        last_log=time.monotonic();e.logger.info('Official %s test cursor=%d/%d; frozen classifier; no training',task,cursor.cursor,n)
        with torch.no_grad():
            while cursor.cursor<n:
                pause_if_requested();start=cursor.cursor;stop=min(start+batch,n)
                if task in ['cloth','books']:
                    with use_cached_vectors():
                        a=link.vectors(gpu,source[start:stop]);b=link.vectors(gpu,target[start:stop]);positive=head(a,b)
                        negatives=link.vectors(gpu,negative[start:stop].ravel()).reshape(stop-start,150,-1)
                        scores=head(a[:,None,:].expand_as(negatives),negatives)
                        values=(1/ranking_metrics(positive,scores)['mrr']).cpu().numpy()
                else:
                    features,_=test.batch(np.arange(start,stop));logits=head(features)
                    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite official logits')
                    values=logits.argmax(1).cpu().numpy()
                    if not ((values>=0)&(values<classes)).all():raise ValueError('Invalid predicted class')
                cursor.commit(start,values);e.metric(dict(step=stop,test_cursor=stop,official_total=n))
                if time.monotonic()-last_log>=30 or stop==n:
                    e.logger.info('Official %s test cursor=%d/%d',task,stop,n);last_log=time.monotonic()
        values=cursor.complete_values()
        if task in ['cloth','books']:
            if not ((values>=1)&(values<=151)).all():raise ValueError('Invalid complete official ranks')
            def metrics(r):return dict(mrr=float(np.mean(1./r)),**{'hits@1':float(np.mean(r<=1)),'hits@3':float(np.mean(r<=3)),'hits@10':float(np.mean(r<=10))})
            result=dict(official=metrics(values),queries=n,reverse_overlap_queries=int(overlap.sum()),nonoverlapping_queries=int((~overlap).sum()),
                reverse_nonoverlap_subset=metrics(values[~overlap]) if (~overlap).any() else None)
            np.savez_compressed(e.directory/'test_link_ranks.npz',source_node=source,target_node=target,rank=values,reverse_overlap=overlap)
            summary.update(test=result)
        else:
            accuracy=float(np.mean(values==test.labels));summary.update(test_accuracy=accuracy,test_loss=None)
            np.savez_compressed(e.directory/'test_predictions.npz',prediction=values,labels=test.labels,
                pairs=test.pairs if test.pairs is not None else np.empty((0,2),dtype=np.int64),
                node_ids=test.indices if test.indices is not None else np.empty(0,dtype=np.int64))
        summary.update(test_evaluations=1,validation_run_directory=str(selected),selection_lock_sha256=fingerprint(lock_file),
            deterministic_test_cursor_complete=True,test_logits_not_retained=True)
        atomic_write_json(e.directory/'summary.json',summary);e.status('completed_resumable_official_test')
    except BaseException as error:e.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error));raise
    finally:e.close()
