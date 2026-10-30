"""Test a validation-locked saved KG classifier once, without retraining it."""
import argparse
import copy
from pathlib import Path

import dgl
import numpy as np
import torch

from .atomic_io import read_json_file, atomic_write_json
from .author_loader import ROOT, fingerprint, verify_originals
from .downstream import ProbeInput, score_probe
from .experiment import Experiment, seed_everything
from .kg_graph_probe import kg_input
from .prepared_data import prepared_directory


def validate_lock(lock, summary, best):
    if (not lock.get('locked_before_test') or lock.get('test_used_for_selection') is not False
            or fingerprint(best) != lock['classifier_sha256']
            or summary['dataset'] != lock['dataset']
            or summary['encoder_sha256'] != lock['encoder_sha256']
            or summary['graph_policy'] != lock['graph_policy']
            or summary.get('feature_transform', 'raw') != lock['feature_transform']
            or summary['test_evaluations'] != 0
            or summary['validation_accuracy'] != lock['selected_validation_accuracy']):
        raise ValueError('Confirmation requires a bound, validation-only selected classifier')


class RawDoubleInput(ProbeInput):
    def batch(self, index):
        features, labels = super().batch(index)
        return features.double(), labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-lock', type=Path, required=True)
    args = parser.parse_args()
    lock = read_json_file(args.selection_lock)
    selected = Path(lock['validation_run_directory'])
    summary = read_json_file(selected/'summary.json')
    best = Path(lock['classifier_checkpoint'])
    validate_lock(lock, summary, best)
    if fingerprint(selected/'summary.json') != lock['validation_summary_sha256']:
        raise ValueError('Validation summary changed after selection')
    verify_originals(); seed_everything(summary['seed']); torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required')
    graph = dgl.graph(([0], [1]), num_nodes=2).to('cuda')
    graph.ndata['x'] = torch.ones(2, 1, device='cuda')
    graph.update_all(dgl.function.copy_u('x', 'm'), dgl.function.sum('m', 'y'))
    if graph.ndata['y'][1].item() != 1:
        raise RuntimeError('DGL CUDA verification failed')
    del graph
    cache = Path(lock['embedding_cache'])
    state = read_json_file(cache/'inference_state.json')
    if (not state['complete'] or state['identity']['checkpoint_sha256'] != lock['encoder_sha256']
            or state['identity']['data']['dataset'] != lock['dataset']
            or fingerprint(cache/'fused_embeddings.npy') != state['output_sha256']):
        raise ValueError('Frozen complete feature cache changed')
    manifest = read_json_file(prepared_directory(lock['dataset'])/'manifest.json')
    for name, digest in manifest['files'].items():
        if fingerprint(prepared_directory(lock['dataset'])/name) != digest:
            raise ValueError('Official prepared data changed')
    embeddings = np.load(cache/'fused_embeddings.npy', mmap_mode='r')
    if embeddings.shape != (manifest['nodes'], 1024):
        raise ValueError('Full-width full-node cache required')
    config = copy.deepcopy(read_json_file(selected/'config.json'))
    config['name'] = 'kg_selected_test_'+lock['dataset']
    config['evaluation']['confirmation'] = {'validation_run':str(selected), 'selection_lock':str(args.selection_lock),
        'selection_lock_sha256':fingerprint(args.selection_lock), 'no_retraining':True}
    audit = copy.deepcopy(state['identity']['data'])
    experiment = Experiment(config, audit, best)
    try:
        saved = torch.load(best, map_location='cpu', weights_only=False)
        double = lock['feature_transform'] == 'train_pair_affine'
        head = torch.nn.Linear(2048, manifest['relations'], dtype=torch.float64 if double else torch.float32).cuda()
        head.load_state_dict(saved['classifier'])
        raw_valid = kg_input(lock['dataset'], embeddings, 'valid')
        raw_test = kg_input(lock['dataset'], embeddings, 'test')
        input_type = RawDoubleInput if double else ProbeInput
        valid = input_type(embeddings, raw_valid.labels, pairs=raw_valid.pairs)
        accuracy, _, _ = score_probe(head, valid)
        if accuracy != summary['validation_accuracy']:
            raise ValueError('Saved selected classifier failed validation replay before testing')
        test = input_type(embeddings, raw_test.labels, pairs=raw_test.pairs)
        accuracy, loss, prediction = score_probe(head, test, save_predictions=True)
        np.savez_compressed(experiment.directory/'test_predictions.npz', prediction=prediction, labels=test.labels, pairs=test.pairs)
        summary = copy.deepcopy(summary)
        summary.update(test_accuracy=accuracy, test_loss=loss, test_evaluations=1,
            validation_run_directory=str(selected), confirmation_without_retraining=True,
            selection_lock_sha256=fingerprint(args.selection_lock), paper_reproduction_complete=False)
        atomic_write_json(experiment.directory/'summary.json', summary)
        experiment.metric({'step':1, 'test_accuracy':accuracy, 'test_loss':loss, 'validation_accuracy':summary['validation_accuracy']})
        experiment.logger.info('Saved validation-selected head: test_accuracy=%.8f; no retraining', accuracy)
        experiment.status('completed_kg_selected_confirmation')
    except BaseException as error:
        experiment.status('failed', error=str(error)); raise
    finally:
        experiment.close()


if __name__ == '__main__':
    main()
