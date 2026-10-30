"""Frozen pilot encoder -> full-neighbor inference -> official-split linear probe.

This tests the complete NC pipeline. It is not joint-pretraining transfer reproduction.
"""
import argparse
import json
from pathlib import Path
import time

import dgl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from .data import load_dataset, modality_features
from .author_loader import ROOT
from .experiment import Experiment, seed_everything, write_json, atomic_checkpoint
from .train import make_model


@torch.no_grad()
def encode_layers(encoder, graph, initial_features, batch_size, log=None):
    """All inbound neighbors per layer; algebraically equivalent to eval-mode full GAT."""
    features = initial_features
    encoder.eval()
    for i, (layer, norm) in enumerate(zip(encoder.gat_layers, encoder.norms)):
        output = torch.empty(graph.num_nodes(), norm.normalized_shape[0], dtype=torch.float32)
        loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()),
                 dgl.dataloading.MultiLayerFullNeighborSampler(1), batch_size=batch_size,
                 shuffle=False, drop_last=False, num_workers=0)
        for inputs, targets, blocks in loader:
            block = blocks[0].to('cuda')
            values = layer(block, features[inputs].cuda()).flatten(1)
            values = F.gelu(norm(values))
            output[targets] = values.cpu()
        features = output
        if log:
            log.info('Completed full-neighbor GAT layer %d/%d nodes=%d', i + 1, len(encoder.gat_layers), graph.num_nodes())
    return features


@torch.no_grad()
def extract_embeddings(model, data, layout, batch_size, log):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    graph = model._prep_graph(data.graph)
    text, image = modality_features(data.features, layout)
    zt = encode_layers(model.gat_t5vit, graph, text, batch_size, log)
    zv = encode_layers(model.gat_clip, graph, image, batch_size, log)
    fused = torch.empty_like(zt)
    for start in range(0, len(zt), batch_size):
        stop = min(start + batch_size, len(zt))
        a, b = zt[start:stop].cuda(), zv[start:stop].cuda()
        gate = model.fuse_gate(torch.cat([a, b], dim=-1))
        mixed = gate * a + (1 - gate) * b
        fused[start:stop] = model.fuse_mlp(torch.cat([mixed, a, b], dim=-1)).cpu()
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available() or min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError('CUDA and positive budgets required')
    torch.set_num_threads(4)
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if config['dataset'] not in ('ele-fashion', 'books-nc'):
        raise ValueError('This command implements only official-split node classification')
    data = load_dataset(config['dataset'], config['feature_layout'])
    if data.audit != checkpoint['data_audit']:
        raise ValueError('Checkpoint and current dataset differ')
    run_config = {**config, 'name': 'probe_' + config['dataset'].replace('-', '_'), 'seed': args.seed, 'stage': 'engineering_pilot_probe',
                  'probe': {'lr': .05, 'optimizer': 'Adam', 'epochs': args.epochs, 'patience': args.patience,
                            'batch_size_inference': args.batch_size, 'seed': args.seed,
                            'inference': 'exact full-neighbor layerwise eval-mode GAT; full NC topology',
                            'selection': 'validation accuracy; test evaluated once after selection',
                            'encoder_checkpoint_steps': checkpoint['step']}}
    experiment = Experiment(run_config, data.audit, args.checkpoint)
    log = experiment.logger
    try:
        log.info('Probe directory: %s; NOT a paper reproduction result', experiment.directory)
        model = make_model(config)
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        started = time.perf_counter()
        embeddings = extract_embeddings(model, data, config['feature_layout'], args.batch_size, log)
        log.info('Extracted %d embeddings, width=%d, seconds=%.2f', len(embeddings), embeddings.shape[1], time.perf_counter() - started)
        model.cpu()
        del model
        torch.cuda.empty_cache()
        torch.save(embeddings, experiment.directory / 'embeddings.pt')
        atomic_checkpoint(experiment.directory / 'checkpoints/inference_metadata.pt', {'node_ids': torch.arange(len(embeddings)), 'graph_policy': 'full official NC topology'})
        features = embeddings.cuda()
        labels = data.labels.cuda()
        indices = {key: values[data.labels[values] >= 0].cuda() for key, values in data.splits.items()}
        for key, value in indices.items():
            if len(value) == 0:
                raise ValueError(f'Empty labeled split: {key}')
        classes = int(labels.max()) + 1
        classifier = torch.nn.Linear(features.shape[1], classes).cuda()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=.05)
        best, best_epoch, stale = -1., 0, 0
        best_state, rows = None, []
        for epoch in range(1, args.epochs + 1):
            classifier.train()
            optimizer.zero_grad()
            logits = classifier(features[indices['train']])
            loss = F.cross_entropy(logits, labels[indices['train']])
            if not torch.isfinite(loss):
                raise FloatingPointError('Probe loss became nonfinite')
            loss.backward()
            optimizer.step()
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(features[indices['val']])
                val_loss = F.cross_entropy(val_logits, labels[indices['val']]).item()
                val_acc = (val_logits.argmax(1) == labels[indices['val']]).float().mean().item()
            values = {'step': epoch, 'train_loss': loss.item(), 'val_loss': val_loss, 'val_accuracy': val_acc, 'lr': .05}
            experiment.metric(values)
            rows.append(values)
            if val_acc > best:
                best, best_epoch, stale = val_acc, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
            else:
                stale += 1
            if epoch % 25 == 0 or epoch == 1:
                log.info('probe epoch=%d train_loss=%.6f val_acc=%.4f best_epoch=%d', epoch, loss.item(), val_acc, best_epoch)
            if stale >= args.patience:
                break
        classifier.load_state_dict(best_state)
        with torch.no_grad():
            predictions = classifier(features[indices['test']]).argmax(1)
            truths = labels[indices['test']]
            test_accuracy = (predictions == truths).double().mean().item()
        torch.save({'classifier': best_state, 'selected_epoch': best_epoch, 'validation_accuracy': best}, experiment.directory / 'checkpoints/probe_best.pt')
        torch.save({'node_ids': indices['test'].cpu(), 'prediction': predictions.cpu(), 'label': truths.cpu()}, experiment.directory / 'test_predictions.pt')
        summary = {'stage': 'engineering_pilot_probe', 'dataset': config['dataset'], 'seed': args.seed,
                   'encoder_frozen': True, 'test_evaluations': 1, 'selected_epoch': best_epoch,
                   'validation_accuracy': best, 'test_accuracy': test_accuracy, 'epochs_executed': len(rows),
                   'paper_reproduction_complete': False, 'deviations': config['deviations'], 'probe_settings': run_config['probe']}
        write_json(experiment.directory / 'summary.json', summary)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        epochs = [row['step'] for row in rows]
        axes[0].plot(epochs, [row['train_loss'] for row in rows], label='train cross entropy')
        axes[0].plot(epochs, [row['val_loss'] for row in rows], label='validation cross entropy')
        axes[0].set(xlabel='Probe epoch', ylabel='Cross entropy (natural log)', title='Frozen CAME pilot: linear probe')
        axes[0].legend(fontsize=8)
        axes[1].plot(epochs, [row['val_accuracy'] for row in rows])
        axes[1].axvline(best_epoch, linestyle='--', color='black', label='Selected epoch')
        axes[1].set(xlabel='Probe epoch', ylabel='Validation accuracy (fraction)', title='Model selection uses validation only')
        axes[1].legend(fontsize=8)
        fig.text(.02, .01, f'Source: {config["dataset"]}, official splits, seed={args.seed}; same-domain limited-step encoder; unverified feature order.', fontsize=8)
        fig.tight_layout(rect=(0, .08, 1, 1))
        fig.savefig(experiment.directory / 'figures/probe_curves.png', dpi=180)
        plt.close(fig)
        matrix = confusion_matrix(truths.cpu().numpy(), predictions.cpu().numpy(), labels=list(range(classes)))
        import numpy as np
        np.save(experiment.directory / 'figures/confusion_matrix.npy', matrix)
        fig, axis = plt.subplots(figsize=(6, 5))
        chart = axis.imshow(matrix, cmap='Blues')
        axis.set(xlabel='Predicted class ID', ylabel='True class ID', title='Official test split: pilot confusion counts')
        fig.colorbar(chart, ax=axis, label='Number of nodes')
        fig.tight_layout()
        fig.savefig(experiment.directory / 'figures/confusion_matrix.png', dpi=180)
        plt.close(fig)
        log.info('Selected epoch=%d validation_accuracy=%.6f test_accuracy=%.6f', best_epoch, best, test_accuracy)
        experiment.status('completed_pilot_probe')
        print(f'RUN_DIR={experiment.directory}', flush=True)
    except BaseException as error:
        log.exception('Probe failed')
        experiment.status('failed', error=str(error))
        raise
    finally:
        experiment.close()


if __name__ == '__main__':
    main()
