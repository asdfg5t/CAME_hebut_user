"""Diagnostic figures with raw numeric artifacts and explicit sample scope."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def training_plots(directory):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / 'logs/metrics.jsonl').read_text().splitlines()]
    if not rows:
        return
    steps = [row['step'] for row in rows]
    config_path = directory / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    stage = config.get('stage', 'engineering_pilot')
    label = 'reconstructed joint protocol; declared assumptions' if stage.startswith('paper_reconstruction') else 'engineering pilot'
    if config.get('sampling', {}).get('pool_policy'):
        label = 'fixed seed subset protocol; declared assumptions'
    caption = f'Source: sampled training subgraphs; seed={config.get("seed")}; {label}. Steps are updates.'
    fig, axis = plt.subplots(figsize=(8, 4))
    for key in ('loss', 'loss_tv', 'loss_tf', 'loss_vf'):
        axis.plot(steps, [row[key] for row in rows], label=key)
    axis.set(xlabel='Optimizer update', ylabel='InfoNCE loss (natural log)', title='CAME training losses')
    axis.legend()
    fig.text(0.04, 0.01, caption, fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(directory / 'figures/training_loss.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(steps, [row['peak_allocated_mib'] for row in rows], label='torch allocated peak')
    axes[0].plot(steps, [row['peak_reserved_mib'] for row in rows], label='torch reserved peak')
    axes[0].set(xlabel='Optimizer update', ylabel='GPU memory (MiB)', title='PyTorch GPU memory')
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, [row['step_seconds'] for row in rows])
    axes[1].set(xlabel='Optimizer update', ylabel='Seconds per update', title='Forward/backward/optimizer time')
    fig.text(.04, .01, caption, fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(directory / 'figures/training_resources.png', dpi=180)
    plt.close(fig)


def embedding_plots(directory, embeddings, routing, nodes, caption):
    directory = Path(directory)
    arrays = {key: value.detach().cpu().float().numpy() for key, value in embeddings.items()}
    np.savez_compressed(directory / 'figures/embedding_diagnostics.npz', node_ids=nodes.numpy(), **arrays)
    (directory / 'figures/diagnostic_scope.txt').write_text(caption, encoding='utf-8')
    if len(nodes) >= 3:
        joined = np.concatenate([arrays[key] for key in ('z1', 'z2', 'z3')])
        projection = PCA(n_components=2, random_state=0).fit_transform(joined)
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for i, name in enumerate(('Text view', 'Image / placeholder view', 'Fused view')):
            part = projection[i * len(nodes):(i + 1) * len(nodes)]
            axis.scatter(part[:, 0], part[:, 1], s=12, alpha=.7, label=name)
        axis.set(xlabel='Joint PCA component 1 (embedding units)', ylabel='Joint PCA component 2 (embedding units)', title='CAME representation diagnostic')
        axis.legend(fontsize=8)
        fig.text(.03, .01, caption, fontsize=7, wrap=True)
        fig.tight_layout(rect=(0, .1, 1, 1))
        fig.savefig(directory / 'figures/embedding_pca.png', dpi=180)
        plt.close(fig)
    if routing is None:
        return
    numeric = {}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for axis, group in zip(axes, ('text', 'image', 'fusion')):
        logits = routing[group]
        k = routing['top_k']
        ids = logits.topk(k, dim=-1).indices.flatten().cpu()
        counts = torch.bincount(ids, minlength=logits.shape[-1]).numpy()
        fractions = counts / counts.sum()
        numeric[group] = fractions.tolist()
        axis.bar(np.arange(1, len(counts) + 1), fractions)
        axis.set(xlabel='Expert ID (1-based)', ylabel='Fraction of selected routes', title=group.capitalize() + ' expert use')
    numeric['modality_weights_mean'] = routing['modality'].softmax(-1).mean(0).cpu().tolist()
    (directory / 'figures/routing_statistics.json').write_text(json.dumps(numeric, indent=2))
    fig.text(.02, .01, caption, fontsize=7)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(directory / 'figures/expert_usage.png', dpi=180)
    plt.close(fig)


def joint_diagnostics(experiment, model, data, sampler, seeds, dataset, config):
    """Bounded eval diagnostic on declared training seeds; no downstream labels used."""
    graph, nodes, _ = sampler.sample(seeds.tolist())
    routes, hooks = {}, []
    has_routes = hasattr(model.fuse_mlp, 'modality_router')
    if has_routes:
        for group in ('text', 'image', 'fusion'):
            def capture(module, inputs, output, key=group):
                routes[key] = output.detach()
            hooks.append(getattr(model.fuse_mlp, group + '_expert_router').register_forward_hook(capture))
        hooks.append(model.fuse_mlp.modality_router.register_forward_hook(
            lambda module, inputs, output: routes.update(modality=output.detach())))
    try:
        model.eval()
        with torch.no_grad():
            text, image = data.modality_batch(nodes)
            _, embeddings = model(graph.to('cuda'), text.cuda(), image.cuda(), return_embeddings=True, compute_loss=False)
        routes['top_k'] = config['model']['top_k']
        caption = f'Source: {dataset}; first {len(seeds)} training seeds, {len(nodes)} PPR-union nodes; seed={config["seed"]}; eval diagnostic, verified feature semantics; no test labels.'
        embedding_plots(experiment.directory, embeddings, routes if has_routes else None, nodes, caption)
    finally:
        for hook in hooks:
            hook.remove()
