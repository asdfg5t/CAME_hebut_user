"""Full-corpus, resumable CUDA training for the two paper pretraining protocols."""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from .author_loader import ROOT
from .prepared_data import load_prepared
from .joint_sampling import PackedPPRSampler, JointSchedule
from .experiment import Experiment, seed_everything, rng_state, restore_rng, atomic_checkpoint, write_json
from .train import make_model
from .plots import training_plots, joint_diagnostics
from .pause_control import TrainingPaused, pause_requested


def validate_joint(config):
    required = {'name', 'stage', 'seed', 'datasets', 'model', 'sampling', 'training', 'deviations'}
    if not required <= set(config) or set(config) - required - {'storage','variant'}:
        raise ValueError('Unexpected/missing joint configuration fields')
    if config['stage'] != 'paper_reconstruction_with_declared_assumptions':
        raise ValueError('Reconstructed protocols must remain explicitly labeled')
    allowed = [['ogbn-products', 'books-lp', 'cloth-copurchase'], ['ogbn-products', 'books-nc', 'cloth-copurchase']]
    if config['datasets'] not in allowed:
        raise ValueError('Joint corpus must match one of the two documented paper protocols')
    m, s, t = config['model'], config['sampling'], config['training']
    if m['hidden_dim'] != 1024 or m['experts_per_view'] != 6 or m['top_k'] != 2 or m['layers'] != 2:
        raise ValueError('Main paper profile must retain the disclosed architecture')
    from .variants import expected_loss_weights
    expected=expected_loss_weights({'w_tv':.5,'w_tf':.1,'w_vf':.5},config.get('variant','came'))
    if (m['w_tv'], m['w_tf'], m['w_vf']) != (expected['w_tv'],expected['w_tf'],expected['w_vf']):
        raise ValueError('Objective weights differ from paper main profile')
    if t['epochs'] != 8 or t['lr'] != .0001 or t['precision'] != 'fp32' or s['top_k'] != 128:
        raise ValueError('Main profile hyperparameters differ from disclosed settings')
    if s['seed_batch_size'] < 1 or t['checkpoint_every'] < 1 or t['max_steps'] is not None and t['max_steps'] < 1:
        raise ValueError('Invalid batch/checkpoint/step budget')
    if set(s['seed_limits']) != set(config['datasets']):
        raise ValueError('Every graph needs an explicit seed budget')
    if any(value is not None and value < 1 for value in s['seed_limits'].values()):
        raise ValueError('Seed budget must be positive or null for full coverage')


def signature(config):
    value = copy.deepcopy(config)
    value['training'].pop('max_steps')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/joint_representation_reconstruction.json')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.setdefault('storage', json.loads((ROOT / 'configs/storage.json').read_text()))
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    if args.seed is not None:
        config['seed'] = args.seed
    validate_joint(config)
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; CPU fallback forbidden')
    torch.set_num_threads(4)
    seed_everything(config['seed'])
    print('Validating complete corpus assets; no incomplete feature arrays are accepted.', flush=True)
    data = {name: load_prepared(name, require_verified=True) for name in config['datasets']}
    audit = {name: value.audit for name, value in data.items()}
    experiment = Experiment(config, audit, args.resume)
    samplers = {}
    model, optimizer, step = None, None, 0
    generator = torch.Generator().manual_seed(config['seed'])
    schedule = JointSchedule({name: value.graph.num_nodes() for name, value in data.items()},
                             config['sampling']['seed_batch_size'], generator, config['sampling']['seed_limits'])
    counts = {name: {'steps': 0, 'seeds': 0, 'skipped': 0} for name in data}
    try:
        model = make_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
        samplers = {name: PackedPPRSampler(value, config['sampling']) for name, value in data.items()}
        if args.resume:
            previous = torch.load(args.resume, map_location='cpu', weights_only=False)
            if previous['schema'] != 'came_joint_v1' or signature(previous['config']) != signature(config) or previous['data_audit'] != audit:
                raise ValueError('Cannot resume changed corpus/config/protocol')
            model.load_state_dict(previous['model'])
            optimizer.load_state_dict(previous['optimizer'])
            schedule.load_state_dict(previous['schedule'])
            counts, step = previous['counts'], previous['step']
            restore_rng(previous['rng'])
        log = experiment.logger
        log.info('Joint run: %s; parameters=%d; corpus=%s', experiment.directory, sum(p.numel() for p in model.parameters()), config['datasets'])
        log.warning('Declared reconstruction assumptions: %s', config['deviations'])

        def save():
            for sampler in samplers.values():
                sampler.flush()
            state = {'schema': 'came_joint_v1', 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'rng': rng_state(), 'schedule': schedule.state_dict(), 'config': config, 'data_audit': audit,
                     'step': step, 'counts': counts}
            atomic_checkpoint(experiment.checkpoint_directory / 'last.pt', state)

        while schedule.epoch < config['training']['epochs']:
            if pause_requested():
                save()
                training_plots(experiment.directory)
                raise TrainingPaused('Pretraining saved at optimizer boundary; step=' + str(step))
            if config['training']['max_steps'] is not None and step >= config['training']['max_steps']:
                break
            selected = schedule.next_batch()
            if selected is None:
                log.info('Completed full declared epoch %d; graph seed coverage=%s', schedule.epoch+1, schedule.cursors)
                schedule.epoch += 1
                if schedule.epoch < config['training']['epochs']:
                    schedule.new_epoch()
                save()
                continue
            name, seeds = selected
            started = time.perf_counter()
            graph, nodes, residual = samplers[name].sample(seeds.tolist())
            sampling_seconds = time.perf_counter() - started
            counts[name]['seeds'] += len(seeds)
            if len(nodes) < 2:
                counts[name]['skipped'] += 1
                log.info('Skipped no-negative subgraph: %s seed_ids=%s', name, seeds.tolist())
                continue
            a, b = data[name].modality_batch(nodes)
            graph = graph.to('cuda')
            a, b = a.cuda(), b.cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            loss, outputs = model(graph, a, b, return_embeddings=True)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite joint loss')
            loss.backward()
            gradients = [value.grad for value in model.parameters() if value.grad is not None]
            if not all(torch.isfinite(value).all() for value in gradients):
                raise FloatingPointError('Nonfinite joint gradient')
            raw_norm = torch.stack([value.float().norm().square() for value in gradients]).sum().sqrt().item()
            limit = config['training']['max_grad_norm']
            if raw_norm > config['training']['gradient_abort_norm']:
                log.warning('Large gradient %.4e dataset=%s clipping=%s', raw_norm, name, limit)
                if limit is None:
                    raise FloatingPointError('Large finite gradient; explicit clipping disabled')
            if limit is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), limit, error_if_nonfinite=True)
            applied = torch.stack([value.float().norm().square() for value in gradients]).sum().sqrt().item()
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            counts[name]['steps'] += 1
            values = {'step': step, 'dataset': name, 'epoch': schedule.epoch+1, 'graph_seed_cursor': schedule.cursors[name],
                      'graph_seed_budget': len(schedule.orders[name]), 'subgraph_nodes': len(nodes), 'subgraph_edges': graph.num_edges(),
                      'loss': loss.item(), 'loss_tv': outputs['loss12'].item(), 'loss_tf': outputs['loss13'].item(), 'loss_vf': outputs['loss23'].item(),
                      'grad_norm': raw_norm, 'applied_grad_norm': applied, 'lr': optimizer.param_groups[0]['lr'],
                      'gradient_clipped': int(limit is not None and raw_norm > limit),
                      'zero_text_embeddings': int((outputs['z1'].norm(dim=1) == 0).sum()),
                      'zero_image_embeddings': int((outputs['z2'].norm(dim=1) == 0).sum()),
                      'sampling_seconds': sampling_seconds, 'step_seconds': time.perf_counter()-started, 'ppr_residual_max': residual,
                      'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20, 'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
            if not all(math.isfinite(value) for value in values.values() if isinstance(value, (float, int))):
                raise FloatingPointError('Nonfinite metric')
            experiment.metric(values)
            log.info('step=%d epoch=%d graph=%s coverage=%d/%d nodes=%d loss=%.6f sample=%.3fs train=%.3fs peak=%.1f MiB',
                     step, schedule.epoch+1, name, schedule.cursors[name], len(schedule.orders[name]), len(nodes), loss.item(), sampling_seconds, values['step_seconds'], values['peak_allocated_mib'])
            del outputs, loss, graph, a, b
            if step == 1 or step % config['training']['checkpoint_every'] == 0:
                save()
                training_plots(experiment.directory)
        save()
        training_plots(experiment.directory)
        if step:
            diagnostic_name = config['datasets'][0]
            joint_diagnostics(experiment, model, data[diagnostic_name], samplers[diagnostic_name],
                              schedule.orders[diagnostic_name][:8], diagnostic_name, config)
        complete = schedule.epoch == config['training']['epochs']
        summary = {'stage': config['stage'], 'step': step, 'completed_epochs': schedule.epoch, 'counts': counts,
                   'checkpoint': str(experiment.checkpoint_directory / 'last.pt'),
                   'seed_coverage_current_epoch': schedule.cursors, 'complete_declared_pretraining': complete,
                   'paper_reproduction_complete': False, 'deviations': config['deviations']}
        write_json(experiment.directory / 'summary.json', summary)
        experiment.status('completed_pretraining' if complete else 'completed_partial_budget', steps=step, completed_epochs=schedule.epoch)
        if complete:
            atomic_checkpoint(experiment.checkpoint_directory / 'encoder_final.pt', {'schema': 'came_encoder_v1',
                              'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                              'config': config, 'data_audit': audit, 'step': step, 'completed_epochs': schedule.epoch})
        print(f'RUN_DIR={experiment.directory}', flush=True)
    except BaseException as error:
        if isinstance(error, TrainingPaused):
            experiment.logger.info('Safe pause: %s', error)
        else:
            experiment.logger.exception('Joint training failed/interrupted')
        experiment.status('paused' if isinstance(error, TrainingPaused) else 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error), steps=step)
        raise
    finally:
        for sampler in samplers.values():
            sampler.close()
        experiment.close()


if __name__ == '__main__':
    main()
