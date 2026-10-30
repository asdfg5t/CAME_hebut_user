"""Single-seed CUDA diagnostic with held-out test and explicit topology candidates."""
import argparse
import copy
import gc
import json
import shutil
from pathlib import Path
import time

import dgl
import numpy as np
import torch
from .author_loader import ROOT, fingerprint, verify_originals
from .downstream import ProbeInput, node_inputs, train_linear_probe, score_probe
from .experiment import Experiment, write_json, seed_everything
from .prepared_data import load_prepared
from .stream_inference import extract_full_embeddings
from .train import make_model
from .pause_control import TrainingPaused
from .storage_budget import size_bytes


class CachedProbeInput(ProbeInput):
    def cache_validation(self, limit_mib, logger):
        required = len(self) * (self.width*4+8)
        if not limit_mib or required > limit_mib*2**20 or required+512*2**20 > torch.cuda.mem_get_info()[0]:
            logger.info('Validation cache not allocated: bytes=%d; CUDA scoring remains enabled', required)
            return
        features = torch.empty((len(self), self.width), dtype=torch.float32, device='cuda')
        labels = torch.empty(len(self), dtype=torch.long, device='cuda')
        for start in range(0, len(self), 4096):
            stop = min(start+4096, len(self))
            features[start:stop], labels[start:stop] = super().batch(np.arange(start, stop))
        self.cuda_features, self.cuda_labels = features, labels
        logger.info('Fixed validation features cached on CUDA: %.1f MiB; same 4096-row scoring blocks', required/2**20)

    def batch(self, index):
        if hasattr(self, 'cuda_features'):
            index = torch.as_tensor(index, dtype=torch.long, device='cuda')
            return self.cuda_features[index], self.cuda_labels[index]
        return super().batch(index)


def owned_cache_path(storage, checkpoint_hash, dataset, policy):
    return Path(storage['embedding_cache_root'])/'quality_pilot'/checkpoint_hash[:20]/(dataset+'_'+policy)


def remove_owned_cache(path, storage):
    path = Path(path).resolve()
    root = (Path(storage['embedding_cache_root'])/'quality_pilot').resolve()
    if not path.is_relative_to(root) or len(path.relative_to(root).parts) != 2:
        raise ValueError('Refusing to remove an unrelated cache')
    if path.is_dir():
        shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['wiki-cs','ogbn-arxiv','books-nc','ele-fashion'], required=True)
    parser.add_argument('--representation', choices=['came','clip_text'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--graph-policy', choices=['original','bidirected'], default='original')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split-id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--inference-batch-size', type=int, default=32)
    parser.add_argument('--validation-cache-mib', type=int, default=400)
    parser.add_argument('--owned-cache-limit-gib', type=float, default=3)
    parser.add_argument('--resume-head', type=Path)
    parser.add_argument('--confirm-test', action='store_true')
    parser.add_argument('--discard-derived-cache', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required')
    verify_originals()
    torch.set_num_threads(4)
    seed_everything(args.seed)
    # Verify actual Torch and DGL CUDA operations even for the feature baseline.
    check = dgl.graph(([0],[1]),num_nodes=2).to('cuda')
    check.ndata['x'] = torch.ones(2,1,device='cuda')
    check.update_all(dgl.function.copy_u('x','m'),dgl.function.sum('m','y'))
    if check.ndata['y'][1].item() != 1:
        raise RuntimeError('DGL CUDA operation failed')
    del check
    storage = json.loads((ROOT/'configs/storage.json').read_text())
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    if args.representation == 'came' and saved is None:
        raise ValueError('CAME diagnostic needs a saved encoder')
    data = load_prepared(args.dataset,build_graph=args.representation=='came')
    data.audit = copy.deepcopy(data.audit)
    if args.graph_policy == 'bidirected':
        if args.dataset not in ('wiki-cs','ogbn-arxiv','books-nc') or args.representation != 'came':
            raise ValueError('Bidirectionality candidates are limited to the three deficient NC graphs')
        before = data.graph.num_edges()
        data.graph = dgl.to_bidirected(data.graph)
        data.audit['diagnostic_topology'] = {'policy':'bidirected_unique_edges_preserve_node_ids',
            'original_edges':before,'candidate_edges':data.graph.num_edges(),
            'author_protocol_confirmed':False,
            'source': {'wiki-cs':'DGL 1.1.2 WikiCSDataset.process',
                       'books-nc':'MM-Graph nc/main.py to_bidirected_with_reverse_mapping',
                       'ogbn-arxiv':'unconfirmed engineering candidate'}[args.dataset]}
    config = copy.deepcopy(saved['config']) if saved else {'stage':'quality_feature_diagnostic','deviations':[]}
    config.update(name='quality_'+args.representation+'_'+args.dataset,seed=args.seed,storage=storage)
    config['evaluation'] = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['evaluation'].update(dataset=args.dataset,selection_split='official_stopping_mask' if args.dataset=='wiki-cs' else 'official_validation',
                                test_policy='single_confirmation_after_validation_lock' if args.confirm_test else 'held_out_not_evaluated')
    config['deviations'] = list(config.get('deviations',[]))+['Small diagnostic; one Wiki-CS split; topology candidates explicitly unconfirmed; not complete paper reproduction.']
    experiment = Experiment(config,data.audit,args.resume_head or args.checkpoint)
    cache = None
    owned = False
    try:
        if saved:
            checkpoint_hash = fingerprint(args.checkpoint)
            model = make_model(config)
            model.load_state_dict(saved['model'])
            del saved
            gc.collect()
            ordinary = Path(storage['embedding_cache_root'])/checkpoint_hash[:20]/args.dataset
            cache = ordinary if args.graph_policy=='original' and (ordinary/'inference_state.json').is_file() else owned_cache_path(storage,checkpoint_hash,args.dataset,args.graph_policy)
            owned = cache != ordinary
            if owned and not (cache/'fused_embeddings.npy').exists():
                required = data.audit['nodes']*1024*4+2**20
                if size_bytes(Path(storage['embedding_cache_root'])/'quality_pilot')+required > args.owned_cache_limit_gib*2**30:
                    raise RuntimeError('Diagnostic embedding limit exceeded; preserve results and review owned caches')
            embeddings = extract_full_embeddings(model,data,cache,checkpoint_hash,args.inference_batch_size,experiment.logger)
            del model
            torch.cuda.empty_cache()
        else:
            embeddings = data.features[:,:512].numpy() if isinstance(data.features,torch.Tensor) else data.features
        train, selection, test = node_inputs(data,embeddings,args.split_id)
        selection = CachedProbeInput(embeddings,selection.labels,indices=selection.indices)
        selection.cache_validation(args.validation_cache_mib,experiment.logger)
        summary = train_linear_probe(train,selection,test,int(np.max(data.labels))+1,args.epochs,args.patience,args.seed,experiment,
                    selection_criterion='loss' if args.dataset=='wiki-cs' else 'accuracy',resume=args.resume_head,evaluate_test=args.confirm_test)
        if args.dataset=='wiki-cs':
            summary['stopping_accuracy'] = summary.pop('validation_accuracy')
            best = torch.load(experiment.checkpoint_directory/'probe_best.pt',map_location='cpu',weights_only=False)
            classifier = torch.nn.Linear(train.width,int(np.max(data.labels))+1).cuda()
            classifier.load_state_dict(best['classifier'])
            ids = np.flatnonzero(data.splits['val_masks'][args.split_id])
            valid = ProbeInput(embeddings,data.labels[ids],indices=ids)
            summary['validation_accuracy'],summary['validation_loss'],predictions = score_probe(classifier,valid,save_predictions=True)
            np.savez_compressed(experiment.directory/'validation_predictions.npz',prediction=predictions,labels=valid.labels,node_ids=ids)
        summary.update(dataset=args.dataset,representation=args.representation,graph_policy=args.graph_policy,
                       split_id=args.split_id,encoder_sha256=fingerprint(args.checkpoint) if args.checkpoint else None,
                       diagnostic_only=True,paper_reproduction_complete=False,feature_cache=str(cache) if cache else 'original_CLIP_view',
                       test_policy=config['evaluation']['test_policy'])
        write_json(experiment.directory/'summary.json',summary)
        experiment.status('completed_quality_probe')
        if args.discard_derived_cache and owned:
            # Close our mapping explicitly; ProbeInput and Wiki validation views
            # can still reference it until this function returns on Windows.
            embeddings._mmap.close()
            del embeddings,train,selection,test
            gc.collect()
            remove_owned_cache(cache,storage)
            write_json(experiment.directory/'cache_eviction.json',{'directory':str(cache),'derived_only':True})
    except BaseException as error:
        experiment.status('paused' if isinstance(error,TrainingPaused) else 'failed',error=str(error))
        raise
    finally:
        experiment.close()


if __name__ == '__main__':
    main()
