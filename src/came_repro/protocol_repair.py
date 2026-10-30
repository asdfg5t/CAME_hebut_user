"""Single-variable graph/text candidates; original datasets and encoders stay immutable."""
import argparse
import copy
import gc
import hashlib
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from . import link_evaluation as link
from .atomic_io import read_json_file
from .author_loader import ROOT, fingerprint, verify_originals
from .experiment import Experiment, atomic_checkpoint, write_json, seed_everything, rng_state, restore_rng
from .head_resume import head_signature
from .kg_graph_probe import training_reverse_graph
from .lp_validation_repair import make_head, TrainOnlyNegatives, load_verified_cache
from .pause_control import TrainingPaused, pause_requested, pause_if_requested
from .prepared_data import load_prepared, prepared_directory
from .remaining_lp_pilot import use_cached_vectors, verify_row_cache_equivalence
from .storage_budget import guard_allocation
from .stream_inference import extract_full_embeddings
from .task_metrics import reverse_overlap_mask
from .train import make_model

PROFILE = ROOT / 'configs/protocol_repair_20261014.json'


def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU fallback')
    graph = dgl.graph(([0], [1]), num_nodes=2).to('cuda')
    graph.ndata['x'] = torch.ones(2, 1, device='cuda')
    graph.update_all(dgl.function.copy_u('x', 'm'), dgl.function.sum('m', 'y'))
    if graph.ndata['y'][1].item() != 1:
        raise RuntimeError('Actual DGL CUDA failed')


def bidirected_train_keys(source, target, n):
    source, target = np.asarray(source, dtype=np.int64), np.asarray(target, dtype=np.int64)
    if source.shape != target.shape or source.ndim != 1 or n <= 0:
        raise ValueError('Aligned train node arrays required')
    if np.any(source < 0) or np.any(target < 0) or np.any(source >= n) or np.any(target >= n):
        raise ValueError('Train node ID outside official graph')
    keep = source != target
    return np.unique(np.concatenate((source[keep]*n+target[keep], target[keep]*n+source[keep])))


def apply_lp_graph(data, source, target):
    actual_source, actual_target = data.graph.edges()
    np.testing.assert_array_equal(actual_source.numpy(), source)
    np.testing.assert_array_equal(actual_target.numpy(), target)
    n = data.graph.num_nodes()
    keys = bidirected_train_keys(source, target, n)
    data.graph = dgl.graph((torch.from_numpy(keys//n), torch.from_numpy(keys % n)), num_nodes=n)
    data.audit['diagnostic_topology'] = {
        'policy': 'upstream_train_bidirected_unique', 'original_edges': len(source),
        'candidate_edges': len(keys), 'canonical_keys_sha256': hashlib.sha256(keys.tobytes()).hexdigest(),
        'derivation': 'unique train edges and reverses; self-loops handled by preserved author model',
        'no_validation_test_edges_added_as_inputs': True, 'author_came_loader_confirmed': False,
        'upstream_revision': '0769fa670c69a6dc0756a2d7fc697b980e494775'}


def candidate_cache(dataset, model_hash, storage):
    policy = 'ofa_prefix_train_reverse' if dataset == 'WN18RR' else 'train_bidirected_unique'
    return Path(storage['embedding_cache_root'])/'protocol_repair'/model_hash[:20]/(dataset+'_'+policy)


def encode_prefixed_text(data, directory, prefix, batch_size):
    """Only the pinned task-independent OFA prefix; no labels or relation text."""
    from transformers import CLIPModel, CLIPTokenizerFast
    text_path = prepared_directory('WN18RR')/'node_text.jsonl'
    import json
    rows = [json.loads(line) for line in text_path.read_text(encoding='utf-8').splitlines()]
    if any(row['node_id'] != i for i, row in enumerate(rows)) or len(rows) != data.graph.num_nodes():
        raise ValueError('Official full entity ordering required')
    texts = [prefix+row['text'] for row in rows]
    text_hash = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    model_directory = ROOT/'data/models/clip_vit_base_patch32'
    manifest = read_json_file(model_directory/'manifest.json')
    weights_hash = fingerprint(model_directory/'pytorch_model.bin')
    if weights_hash != manifest['files']['pytorch_model.bin']['sha256']:
        raise ValueError('CLIP model weights changed')
    identity = {'node_text_sha256': fingerprint(text_path), 'prefixed_text_sha256': text_hash,
                'prefix': prefix, 'CLIP_revision': manifest['revision'], 'weights_sha256': weights_hash,
                'normalization': 'L2', 'max_length': 77, 'precision': 'fp32', 'nodes': len(texts)}
    directory.mkdir(parents=True, exist_ok=True)
    path, state_path = directory/'prefixed_clip_text.npy', directory/'text_encoding_state.json'
    state = read_json_file(state_path) if state_path.exists() else {'identity': identity, 'cursor': 0, 'complete': False}
    if state['identity'] != identity:
        raise ValueError('Prefix cache identity changed')
    if state['complete']:
        if fingerprint(path) != state['output_sha256']:
            raise ValueError('Complete prefix cache changed')
    else:
        if state['cursor'] and not path.exists():
            raise ValueError('Prefix cursor has no persistent feature array')
        if not path.exists():
            guard_allocation(path, len(texts)*512*4+1024**2)
        output = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=np.float32, shape=(len(texts), 512))
        encoder = CLIPModel.from_pretrained(model_directory, local_files_only=True).cuda().eval()
        tokenizer = CLIPTokenizerFast.from_pretrained(model_directory, local_files_only=True)
        for start in range(state['cursor'], len(texts), batch_size):
            stop = min(start+batch_size, len(texts))
            inputs = tokenizer(texts[start:stop], padding='max_length', truncation=True, max_length=77, return_tensors='pt').to('cuda')
            with torch.no_grad():
                values = F.normalize(encoder.get_text_features(**inputs), dim=-1)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite prefixed CLIP vectors')
            output[start:stop] = values.cpu().numpy(); output.flush()
            state['cursor'] = stop; write_json(state_path, state); pause_if_requested()
        state.update(complete=True, output_sha256=fingerprint(path)); write_json(state_path, state)
        del encoder, output; gc.collect(); torch.cuda.empty_cache()
    data.features = np.load(path, mmap_mode='r')
    data.audit['text_candidate'] = dict(identity, output_sha256=state['output_sha256'], author_prefix_confirmed=False)
    return data


def encode(args, profile):
    model_hash = fingerprint(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = copy.deepcopy(saved['config']); storage = read_json_file(ROOT/'configs/storage.json')
    cache = candidate_cache(args.dataset, model_hash, storage)
    data = load_prepared(args.dataset)
    if args.dataset == 'WN18RR':
        data.graph = training_reverse_graph(data.graph)
        data.audit['diagnostic_topology'] = {'policy': 'train_edges_plus_reverse_preserve_parallel_edges',
            'no_validation_test_edges_added_as_inputs': True, 'author_came_loader_confirmed': False}
        data.audit['requested_text_candidate'] = {'prefix': profile['WN']['prefix'],
            'source': 'OFA 9045ccdf data/KG/gen_data.py:84', 'author_prefix_confirmed': False}
    else:
        raw = ROOT/'data/raw/mm_graph'/args.dataset/'lp-edge-split.pt'
        if fingerprint(raw) != data.audit['raw_sha256']['lp-edge-split.pt']:
            raise ValueError('Official raw split changed')
        split = torch.load(str(raw), map_location='cpu', weights_only=False, mmap=True)
        apply_lp_graph(data, np.asarray(split['train']['source_node']), np.asarray(split['train']['target_node']))
        del split
    config.update(name='protocol_encode_'+args.dataset, seed=0, storage=storage)
    config['evaluation'] = {'dataset': args.dataset, 'task': 'encoding', 'profile_sha256': fingerprint(PROFILE),
                            'test_used_for_selection': False, 'engineering_candidate': True}
    config['deviations'] = list(config.get('deviations', []))+['One upstream-supported graph or text candidate; CAME author exact loader remains unreleased.']
    experiment = Experiment(config, data.audit, args.checkpoint)
    try:
        if args.dataset == 'WN18RR':
            experiment.logger.info('CLIP exact OFA prefix encoding; full entities; cursor persisted per batch')
            data = encode_prefixed_text(data, cache, profile['WN']['prefix'], profile['WN']['clip_batch_size'])
            write_json(experiment.directory/'candidate_feature_audit.json', data.audit['text_candidate'])
        model = make_model(config); model.load_state_dict(saved['model']); del saved
        embeddings = extract_full_embeddings(model, data, cache, model_hash, 32, experiment.logger)
        if embeddings.shape != (data.graph.num_nodes(), 1024):
            raise ValueError('Full nodes and 1024 width required')
        write_json(experiment.directory/'summary.json', {'dataset': args.dataset, 'encoder_sha256': model_hash,
            'embedding_cache': str(cache), 'inference_state_sha256': fingerprint(cache/'inference_state.json'),
            'data_audit': data.audit, 'test_evaluations': 0, 'paper_reproduction_complete': False})
        experiment.status('completed_protocol_encoding')
    except BaseException as error:
        experiment.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error)); raise
    finally:
        experiment.close()


def train_lp(gpu, train, valid, recipe, experiment, resume=None):
    seed_everything(0); random = np.random.default_rng(0)
    source, target = (np.asarray(train[k], dtype=np.int64) for k in ('source_node', 'target_node'))
    sampler = TrainOnlyNegatives(source, target, len(gpu))
    overlap = reverse_overlap_mask(source, target, np.asarray(valid['source_node']), np.asarray(valid['target_node']), len(gpu))
    head = make_head(recipe['head_variant']).cuda(); optimizer = torch.optim.Adam(head.parameters(), lr=recipe['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, recipe['step_lr_size'], recipe['step_lr_gamma']) if recipe['step_lr_size'] else None
    best, selected, stale, first, history, best_state = -1., 0, 0, 1, [], None
    if resume:
        old = torch.load(resume, map_location='cpu', weights_only=False)
        if old['schema'] != 'came_lp_protocol_v1' or head_signature(old['config']) != head_signature(experiment.config) or old['data_audit'] != experiment.data_audit:
            raise ValueError('Protocol resume identity changed')
        head.load_state_dict(old['head']); optimizer.load_state_dict(old['optimizer'])
        if scheduler: scheduler.load_state_dict(old['scheduler'])
        restore_rng(old['rng']); random.bit_generator.state = old['sampling_generator']
        best, selected, stale, first, history, best_state = old['best_mrr'], old['selected_epoch'], old['stale'], old['next_epoch'], old['history_rows'], old['best_state']
        for row in history: experiment.metric(row)
    for epoch in range(first, first if stale >= recipe['patience'] else recipe['max_epochs']+1):
        head.train(); order = random.permutation(len(source)); total = 0.; used_lr = optimizer.param_groups[0]['lr']
        for start in range(0, len(order), recipe['batch_size']):
            ids = order[start:start+recipe['batch_size']]; negative = sampler.sample(source[ids], random)
            a, b, c = (link.vectors(gpu, x) for x in (source[ids], target[ids], negative))
            optimizer.zero_grad(set_to_none=True); positive, negative_score = head(a, b), head(a, c)
            if recipe['loss_reduction'] == 'separate_mean_sum':
                loss = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))+F.binary_cross_entropy_with_logits(negative_score, torch.zeros_like(negative_score))
            else:
                loss = F.binary_cross_entropy_with_logits(torch.cat((positive, negative_score)), torch.cat((torch.ones_like(positive), torch.zeros_like(negative_score))))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite LP loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in head.parameters()): raise FloatingPointError('Nonfinite LP gradient')
            optimizer.step(); total += loss.item()*len(ids)
        if scheduler: scheduler.step()
        metric = link.evaluate_split(head, gpu, valid, 64, overlap)['official']['mrr']
        row = {'step': epoch, 'train_loss': total/len(source), 'validation_mrr': metric, 'lr_used': used_lr, 'lr_next': optimizer.param_groups[0]['lr']}
        history.append(row); experiment.metric(row)
        if metric > best:
            best, selected, stale = metric, epoch, 0; best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else: stale += 1
        experiment.logger.info('Protocol LP epoch=%d loss=%.6f validation_MRR=%.8f selected=%d', epoch, row['train_loss'], metric, selected)
        atomic_checkpoint(experiment.checkpoint_directory/'link_head_last.pt', {'schema':'came_lp_protocol_v1',
            'head':head.state_dict(), 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict() if scheduler else None,
            'rng':rng_state(), 'sampling_generator':random.bit_generator.state, 'completed_epochs':epoch, 'next_epoch':epoch+1,
            'best_mrr':best, 'selected_epoch':selected, 'stale':stale, 'best_state':best_state, 'history_rows':history,
            'cursor_policy':'full official train-positive epoch boundary', 'config':experiment.config, 'data_audit':experiment.data_audit})
        if pause_requested(): raise TrainingPaused('Protocol head checkpoint saved')
        if stale >= recipe['patience']: break
    head.load_state_dict(best_state)
    replay = link.evaluate_split(head, gpu, valid, 64, overlap, experiment.directory/'validation_link_ranks.npz')
    if abs(replay['official']['mrr']-best) > 1e-12: raise ValueError('Best validation replay mismatch')
    atomic_checkpoint(experiment.checkpoint_directory/'link_head_best.pt', {'head':best_state, 'selected_epoch':selected, 'validation_mrr':best})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['step'] for r in history], [r['train_loss'] for r in history]); axes[0].set(xlabel='Full train-positive epoch', ylabel='Binary cross entropy')
    axes[1].plot([r['step'] for r in history], [r['validation_mrr'] for r in history]); axes[1].axvline(selected, ls='--', color='black'); axes[1].set(xlabel='Full train-positive epoch', ylabel='Official validation MRR')
    fig.text(.02,.01,experiment.config['evaluation']['dataset']+'; train-only bidirected unique candidate; seed0; test held out.', fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(experiment.directory/'figures/protocol_lp_curves.png', dpi=180);plt.close(fig)
    return dict(recipe, selected_epoch=selected, epochs_executed=len(history), validation_mrr=best, test=None, test_evaluations=0)


def validate_lp(args, profile):
    model_hash = fingerprint(args.checkpoint); storage = read_json_file(ROOT/'configs/storage.json')
    cache = Path(storage['embedding_cache_root'])/model_hash[:20]/args.dataset if args.directed_control else candidate_cache(args.dataset, model_hash, storage)
    state = read_json_file(cache/'inference_state.json')
    policy = state['identity']['data'].get('diagnostic_topology', {}).get('policy')
    if not args.directed_control and policy != 'upstream_train_bidirected_unique':
        raise ValueError('Declared train-only graph candidate required')
    if args.directed_control and (args.dataset != 'cloth-copurchase' or policy is not None):
        raise ValueError('Matched control requires the historical original Cloth cache')
    gpu, audit = load_verified_cache(args.dataset, cache, model_hash)
    split = torch.load(str(ROOT/'data/raw/mm_graph'/args.dataset/'lp-edge-split.pt'), map_location='cpu', weights_only=False, mmap=True)
    config = copy.deepcopy(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['config'])
    recipe = profile['LP'][args.dataset]
    config.update(name='protocol_lp_'+args.dataset, seed=0, storage=storage)
    config['evaluation'] = dict(recipe, dataset=args.dataset, task='link', encoder_sha256=model_hash,
        embedding_cache=str(cache), inference_state_sha256=fingerprint(cache/'inference_state.json'),
        graph_policy='original_directed_matched_control' if args.directed_control else 'upstream_train_bidirected_unique',
        test_used_for_selection=False, profile_sha256=fingerprint(PROFILE))
    experiment = Experiment(config, audit, args.resume_head or args.checkpoint)
    try:
        write_json(experiment.directory/'cuda_row_cache_equivalence.json', verify_row_cache_equivalence()); seed_everything(0)
        with use_cached_vectors(): summary = train_lp(gpu, split['train'], split['valid'], recipe, experiment, args.resume_head)
        summary.update(dataset=args.dataset, task='link', representation='came', seed=0, encoder_sha256=model_hash,
            graph_policy=config['evaluation']['graph_policy'], embedding_cache=str(cache), diagnostic_only=True, paper_reproduction_complete=False)
        write_json(experiment.directory/'summary.json', summary); experiment.status('completed_lp_protocol_validation')
    except BaseException as error:
        experiment.status('paused' if isinstance(error, TrainingPaused) else 'failed', error=str(error)); raise
    finally: experiment.close()


def confirm_lp(args):
    lock = read_json_file(args.selection_lock); run = Path(lock['validation_run_directory']); summary = read_json_file(run/'summary.json')
    from .lp_selected_confirmation import validate_lock
    validate_lock(lock, summary)
    cache = Path(lock['embedding_cache'])
    if fingerprint(run/'summary.json') != lock['validation_summary_sha256'] or fingerprint(cache/'inference_state.json') != lock['inference_state_sha256']:
        raise ValueError('Locked validation or graph cache changed')
    gpu, audit = load_verified_cache(lock['dataset'], cache, lock['encoder_sha256'])
    if audit.get('diagnostic_topology', {}).get('policy') != 'upstream_train_bidirected_unique': raise ValueError('Locked candidate graph required')
    raw = ROOT/'data/raw/mm_graph'/lock['dataset']/'lp-edge-split.pt'
    if fingerprint(raw) != audit['raw_sha256']['lp-edge-split.pt']: raise ValueError('Official raw split changed')
    split = torch.load(str(raw), map_location='cpu', weights_only=False, mmap=True)
    source, target = (np.asarray(split['train'][k]) for k in ('source_node', 'target_node'))
    overlap = {k:reverse_overlap_mask(source, target, np.asarray(split[k]['source_node']), np.asarray(split[k]['target_node']), len(gpu)) for k in ('valid','test')}
    config = read_json_file(run/'config.json');config['name'] = 'protocol_confirm_'+lock['dataset']
    config['evaluation']['selection_lock_sha256'] = fingerprint(args.selection_lock)
    experiment = Experiment(config, audit, Path(lock['classifier_checkpoint']))
    try:
        head = make_head(lock['head_variant']).cuda(); head.load_state_dict(torch.load(lock['classifier_checkpoint'],map_location='cpu',weights_only=False)['head'])
        with use_cached_vectors():
            valid = link.evaluate_split(head, gpu, split['valid'], 64, overlap['valid'])
            if abs(valid['official']['mrr']-summary['validation_mrr']) > 1e-12: raise ValueError('Locked validation replay changed')
            test = link.evaluate_split(head, gpu, split['test'], 64, overlap['test'], experiment.directory/'test_link_ranks.npz')
        summary.update(test=test, test_evaluations=1, validation_run_directory=str(run), confirmation_without_retraining=True,
            selection_lock_sha256=fingerprint(args.selection_lock))
        np.savez_compressed(experiment.directory/'reverse_overlap_queries.npz', **{k+'_excluded_query_indices':np.flatnonzero(v) for k,v in overlap.items()})
        write_json(experiment.directory/'summary.json',summary);experiment.metric({'step':1,'validation_mrr':summary['validation_mrr'],'test_mrr':test['official']['mrr']})
        experiment.status('completed_lp_protocol_confirmation')
    except BaseException as error:
        experiment.status('failed', error=str(error));raise
    finally: experiment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['encode','validate_lp','confirm_lp'], required=True)
    parser.add_argument('--dataset', choices=['sports-copurchase','cloth-copurchase','WN18RR'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume-head', type=Path)
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--directed-control', action='store_true')
    args = parser.parse_args();verify_originals();seed_everything(0);torch.set_num_threads(4);check_cuda()
    profile = read_json_file(PROFILE)
    if args.stage == 'confirm_lp': confirm_lp(args);return
    if args.checkpoint is None or fingerprint(args.checkpoint) != profile['encoder_sha256']: raise ValueError('Preserved frozen encoder required')
    if args.stage == 'encode': encode(args,profile)
    else: validate_lp(args,profile)


if __name__ == '__main__': main()
