"""One complete official LP test using a prelocked validation-selected saved head."""
import argparse
import copy
from pathlib import Path
import dgl
import numpy as np
import torch
from . import link_evaluation as link
from .atomic_io import read_json_file
from .author_loader import ROOT, fingerprint, verify_originals
from .experiment import Experiment, seed_everything, write_json
from .lp_validation_repair import load_verified_cache, make_head
from .remaining_lp_pilot import use_cached_vectors
from .task_metrics import reverse_overlap_mask


def validate_lock(lock,summary):
    if (not lock.get('locked_before_test') or lock.get('test_used_for_selection') is not False
        or summary['test_evaluations']!=0 or summary['dataset']!=lock['dataset']
        or summary['encoder_sha256']!=lock['encoder_sha256'] or summary['head_variant']!=lock['head_variant']
        or summary['validation_mrr']!=lock['selected_validation_mrr']
        or fingerprint(lock['classifier_checkpoint'])!=lock['classifier_sha256']):
        raise ValueError('LP confirmation requires a bound validation-only selected classifier')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selection-lock',type=Path,required=True)
    args=parser.parse_args();verify_originals();seed_everything(0);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
    lock=read_json_file(args.selection_lock);selected=Path(lock['validation_run_directory'])
    summary=read_json_file(selected/'summary.json');validate_lock(lock,summary)
    if fingerprint(selected/'summary.json')!=lock['validation_summary_sha256']:raise ValueError('Selected validation summary changed')
    g=dgl.graph(([0],[1]),num_nodes=2).to('cuda');g.ndata['x']=torch.ones(2,1,device='cuda')
    g.update_all(dgl.function.copy_u('x','m'),dgl.function.sum('m','y'))
    if g.ndata['y'][1].item()!=1:raise RuntimeError('Actual DGL CUDA failed')
    del g
    gpu,audit=load_verified_cache(lock['dataset'],Path(lock['embedding_cache']),lock['encoder_sha256'])
    raw=ROOT/'data/raw/mm_graph'/lock['dataset']/'lp-edge-split.pt'
    if fingerprint(raw)!=audit['raw_sha256']['lp-edge-split.pt']:raise ValueError('Official raw split changed')
    split=torch.load(raw,map_location='cpu',weights_only=False)
    source,target=(np.asarray(split['train'][k]) for k in ('source_node','target_node'))
    overlap={key:reverse_overlap_mask(source,target,np.asarray(split[key]['source_node']),np.asarray(split[key]['target_node']),len(gpu)) for key in ('valid','test')}
    config=copy.deepcopy(read_json_file(selected/'config.json'));config['name']='lp_saved_confirmation_'+lock['dataset']
    config['evaluation']['confirmation']={'validation_run':str(selected),'selection_lock_sha256':fingerprint(args.selection_lock),'retraining':False}
    experiment=Experiment(config,audit,Path(lock['classifier_checkpoint']))
    try:
        head=make_head(lock['head_variant']).cuda()
        head.load_state_dict(torch.load(lock['classifier_checkpoint'],map_location='cpu',weights_only=False)['head'])
        with use_cached_vectors():
            valid=link.evaluate_split(head,gpu,split['valid'],64,overlap['valid'])
            if abs(valid['official']['mrr']-summary['validation_mrr'])>1e-12:raise ValueError('Selected validation replay changed before testing')
            test=link.evaluate_split(head,gpu,split['test'],64,overlap['test'],experiment.directory/'test_link_ranks.npz')
        summary=copy.deepcopy(summary);summary.update(test=test,test_evaluations=1,
            validation_run_directory=str(selected),confirmation_without_retraining=True,selection_lock_sha256=fingerprint(args.selection_lock))
        np.savez_compressed(experiment.directory/'reverse_overlap_queries.npz',**{key+'_excluded_query_indices':np.flatnonzero(v) for key,v in overlap.items()})
        write_json(experiment.directory/'summary.json',summary)
        experiment.metric({'step':1,'validation_mrr':summary['validation_mrr'],'test_mrr':test['official']['mrr'],
            'test_hits1':test['official']['hits@1'],'test_hits3':test['official']['hits@3'],'test_hits10':test['official']['hits@10']})
        experiment.logger.info('Saved LP head confirmation: official_test_MRR=%.8f; no retraining',test['official']['mrr'])
        experiment.status('completed_lp_selected_confirmation')
    except BaseException as error:
        experiment.status('failed',error=str(error));raise
    finally:experiment.close()


if __name__=='__main__':main()
