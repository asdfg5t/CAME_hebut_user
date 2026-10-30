"""CUDA-only, auditable pilot of the preserved CAME model on official real data."""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

from .author_loader import ROOT, load_author_model
os.environ.setdefault('DGLBACKEND', 'pytorch')
os.environ.setdefault('DGLDEFAULTDIR', str(ROOT / '.runtime/dgl'))
import torch
from .data import load_dataset, PPRSampler, modality_features
from .experiment import Experiment, seed_everything, rng_state, restore_rng, atomic_checkpoint, write_json
from .plots import training_plots, embedding_plots


def validate(config):
    expected = {'name', 'stage', 'seed', 'dataset', 'feature_layout', 'model', 'sampling', 'training', 'deviations'}
    if set(config) != expected:
        raise ValueError(f'Unexpected/missing configuration fields: {set(config) ^ expected}')
    sections = {
        'model': {'hidden_dim', 'heads', 'layers', 'dropout', 'experts_per_view', 'top_k', 'temperature', 'w_tv', 'w_tf', 'w_vf'},
        'sampling': {'top_k', 'alpha', 'epsilon', 'max_pushes', 'seed_batch_size', 'seed_limit', 'direction'},
        'training': {'epochs', 'max_steps', 'lr', 'weight_decay', 'checkpoint_every', 'precision'},
    }
    for name, fields in sections.items():
        optional = {'max_grad_norm', 'gradient_abort_norm'} if name == 'training' else set()
        if not fields <= set(config[name]) or set(config[name]) - fields - optional:
            raise ValueError(f'Unexpected/missing {name} configuration fields')
    if config['stage'] != 'engineering_pilot':
        raise ValueError('Formal reproduction is gated on resolving the protocol gaps; use engineering_pilot')
    m, s, t = config['model'], config['sampling'], config['training']
    if m['hidden_dim'] % m['heads'] or not 1 <= m['top_k'] <= m['experts_per_view']:
        raise ValueError('Invalid attention/expert dimensions')
    if t['precision'] != 'fp32' or t['epochs'] < 1 or t['checkpoint_every'] < 1 or t['lr'] <= 0:
        raise ValueError('Unsupported precision or invalid training settings')
    if s['top_k'] < 2 or s['seed_batch_size'] < 1 or s['seed_limit'] < 1:
        raise ValueError('Invalid sampling budget')
    if t['max_steps'] is not None and t['max_steps'] < 1:
        raise ValueError('max_steps must be positive or null')
    if t.get('max_grad_norm') is not None and t['max_grad_norm'] <= 0:
        raise ValueError('max_grad_norm must be positive or null')
    if t.get('gradient_abort_norm', 10000.) <= 0:
        raise ValueError('gradient_abort_norm must be positive')


def resume_signature(config):
    value = copy.deepcopy(config)
    value['training'].pop('max_steps')
    value['training'].setdefault('max_grad_norm', None)
    value['training'].setdefault('gradient_abort_norm', 10000.)
    return value


def make_model(config):
    m = config['model']
    from .variants import model_class,adapt_instance
    author=load_author_model()
    variant=config.get('variant','came')
    model=model_class(author,variant)(512, 512, hidden_1024=m['hidden_dim'], out_1024=m['hidden_dim'],
                               num_heads=m['heads'], dropout=m['dropout'], attn_drop=m['dropout'],
                               num_experts=m['experts_per_view'] * 3, num_selected_experts=m['top_k'],
                               text_gat_layers=m['layers'], vision_gat_layers=m['layers'],
                               temperature=m['temperature'], w12=m['w_tv'], w13=m['w_tf'], w23=m['w_vf'],
                               loss_batch_size=None, use_gate=True)
    return adapt_instance(model,author,variant).cuda()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pilot_ele_fashion_cuda.json')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    config['training'].setdefault('max_grad_norm', None)
    config['training'].setdefault('gradient_abort_norm', 10000.)
    validate(config)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(4)
    seed_everything(config['seed'])
    data = load_dataset(config['dataset'], config['feature_layout'])
    sampler = PPRSampler(data, config['sampling'])
    experiment = Experiment(config, data.audit, args.resume)
    log = experiment.logger
    model, optimizer = None, None
    step, epoch, cursor, skipped = 0, 0, 0, 0
    order = torch.randperm(data.graph.num_nodes())[:config['sampling']['seed_limit']]
    try:
        model = make_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
        if args.resume:
            previous = torch.load(args.resume, map_location='cpu', weights_only=False)
            if resume_signature(config) != resume_signature(previous['config']) or data.audit != previous['data_audit']:
                raise ValueError('Resume configuration/data differ; start a new experiment instead')
            model.load_state_dict(previous['model'])
            optimizer.load_state_dict(previous['optimizer'])
            step, epoch, cursor, order, skipped = (previous[key] for key in ('step', 'epoch', 'cursor', 'order', 'skipped'))
            restore_rng(previous['rng'])
        log.info('Run directory: %s', experiment.directory)
        log.info('CUDA=%s parameters=%d stage=%s; feature order verified=%s', torch.cuda.get_device_name(0),
                 sum(p.numel() for p in model.parameters()), config['stage'], data.audit['layout_verified'])
        log.warning('Declared deviations: %s', config['deviations'])
        last_batch = None

        def save():
            if model is not None and optimizer is not None:
                state = {'schema': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'rng': rng_state(), 'config': config, 'data_audit': data.audit,
                         'step': step, 'epoch': epoch, 'cursor': cursor, 'order': order, 'skipped': skipped}
                atomic_checkpoint(experiment.directory / 'checkpoints/last.pt', state)

        while epoch < config['training']['epochs']:
            if config['training']['max_steps'] is not None and step >= config['training']['max_steps']:
                break
            if cursor >= len(order):
                epoch += 1
                if epoch < config['training']['epochs']:
                    order = torch.randperm(data.graph.num_nodes())[:config['sampling']['seed_limit']]
                    cursor = 0
                continue
            batch = order[cursor:cursor + config['sampling']['seed_batch_size']]
            subgraph, nodes, residual = sampler.sample(batch.tolist())
            if len(nodes) < 2:
                cursor += len(batch)
                skipped += 1
                log.info('Skip singleton PPR subgraph seed_ids=%s (InfoNCE has no negative)', batch.tolist())
                continue
            text, image = modality_features(data.features[nodes], config['feature_layout'])
            graph_cuda = subgraph.to('cuda')
            text, image = text.cuda(), image.cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            loss, outputs = model(graph_cuda, text, image, return_embeddings=True)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(value).all() for value in gradients):
                raise FloatingPointError('Missing/nonfinite gradients')
            grad_norm = torch.stack([value.float().norm() ** 2 for value in gradients]).sum().sqrt().item()
            limit = config['training']['max_grad_norm']
            abort = config['training']['gradient_abort_norm']
            if grad_norm > abort:
                log.warning('Abnormally large pre-clip gradient %.4e; zero embeddings text=%d image=%d; clip=%s',
                            grad_norm, int((outputs['z1'].norm(dim=1) == 0).sum()),
                            int((outputs['z2'].norm(dim=1) == 0).sum()), limit)
                if limit is None:
                    raise FloatingPointError('Large finite gradient; clipping is disabled. Review zero embeddings/attention dropout before updating.')
            if limit is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), limit, error_if_nonfinite=True)
            applied_grad_norm = torch.stack([value.float().norm() ** 2 for value in gradients]).sum().sqrt().item()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            cursor += len(batch)
            step += 1
            values = {'step': step, 'epoch': epoch + 1, 'seed_cursor': cursor, 'seed_budget': len(order),
                      'subgraph_nodes': len(nodes), 'subgraph_edges': subgraph.num_edges(), 'ppr_residual_max': residual,
                      'loss': loss.item(), 'loss_tv': outputs['loss12'].item(), 'loss_tf': outputs['loss13'].item(),
                      'loss_vf': outputs['loss23'].item(), 'grad_norm': grad_norm,
                      'applied_grad_norm': applied_grad_norm, 'gradient_clipped': int(limit is not None and grad_norm > limit),
                      'zero_text_embeddings': int((outputs['z1'].norm(dim=1) == 0).sum()),
                      'zero_image_embeddings': int((outputs['z2'].norm(dim=1) == 0).sum()),
                      'lr': optimizer.param_groups[0]['lr'], 'step_seconds': elapsed,
                      'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                      'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
            if not all(math.isfinite(value) for value in values.values()):
                raise FloatingPointError('Nonfinite metric')
            experiment.metric(values)
            log.info('step=%d epoch=%d seeds=%d/%d nodes=%d loss=%.6f grad=%.4f GPU_peak=%.1f MiB time=%.3fs',
                     step, epoch + 1, cursor, len(order), len(nodes), loss.item(), grad_norm, values['peak_allocated_mib'], elapsed)
            last_batch = (subgraph, nodes)
            del outputs, loss, graph_cuda, text, image
            if step % config['training']['checkpoint_every'] == 0:
                save()
        save()
        training_plots(experiment.directory)
        if last_batch:
            diagnostic_seeds = order[:min(cursor, 8)]
            subgraph, nodes, _ = sampler.sample(diagnostic_seeds.tolist())
            routes = {}
            hooks = []
            for group in ('text', 'image', 'fusion'):
                def capture(module, inputs, output, key=group):
                    routes[key] = output.detach()
                hooks.append(getattr(model.fuse_mlp, group + '_expert_router').register_forward_hook(capture))
            hooks.append(model.fuse_mlp.modality_router.register_forward_hook(lambda module, inputs, output: routes.update(modality=output.detach())))
            try:
                model.eval()
                with torch.no_grad():
                    text, image = modality_features(data.features[nodes], config['feature_layout'])
                    _, embeddings = model(subgraph.to('cuda'), text.cuda(), image.cuda(), return_embeddings=True, compute_loss=False)
                routes['top_k'] = config['model']['top_k']
                caption = f'Source: {config["dataset"]}, union of first {len(diagnostic_seeds)} sampled training seeds ({len(nodes)} nodes), seed={config["seed"]}; unverified modality order; pilot only.'
                embedding_plots(experiment.directory, embeddings, routes, nodes, caption)
            finally:
                for hook in hooks:
                    hook.remove()
        write_json(experiment.directory / 'summary.json', {'stage': config['stage'], 'steps': step, 'epoch_index': epoch,
                    'seed_cursor': cursor, 'seed_budget': len(order), 'skipped_singletons': skipped,
                    'paper_reproduction_complete': False, 'deviations': config['deviations']})
        if step == 0:
            raise RuntimeError('No optimizer updates completed')
        experiment.status('completed_pilot', steps=step)
        print(f'RUN_DIR={experiment.directory}', flush=True)
    except BaseException as error:
        log.exception('Training interrupted or failed')
        experiment.status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error), steps=step)
        # Last checkpoint remains the most recent consistent completed optimizer update.
        try:
            training_plots(experiment.directory)
        except Exception:
            log.exception('Could not render partial curves')
        raise
    finally:
        experiment.close()


if __name__ == '__main__':
    main()
