"""Fixed subset rounds and strict offline PPR using the unchanged author CUDA trainer.

Paper p16 explicitly specifies Products subset pretraining and offline PPR. Exact
pool sizes/selection are unreleased; local pools for the other domains are declared
capacity adaptations. A subset round is never recorded as a full-graph epoch.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from . import joint_train
from .author_loader import ROOT, fingerprint
from .experiment import Experiment, atomic_checkpoint, write_json
from .joint_sampling import JointSchedule, PackedPPRSampler
from .pause_control import TrainingPaused, pause_requested

CURRENT_EXPERIMENT = None
CURRENT_SCHEDULE = None


class FixedSubsetSchedule(JointSchedule):
    def __init__(self, *args, **kwargs):
        global CURRENT_SCHEDULE
        super().__init__(*args, **kwargs)
        CURRENT_SCHEDULE = self
        if CURRENT_EXPERIMENT is not None:
            directory = CURRENT_EXPERIMENT.directory / 'sampling'
            directory.mkdir(exist_ok=True)
            for name, values in self.pools.items():
                np.save(directory / (name + '_seed_pool.npy'), values.numpy())
            write_json(directory / 'pool_manifest.json', {
                'scope': 'Fixed unlabeled seed pools on immutable FULL graph topology; no label-based selection',
                'files_sha256': {path.name: fingerprint(path) for path in directory.glob('*.npy')}})

    def new_epoch(self):
        if not hasattr(self, 'pools'):
            pool_generator = torch.Generator().manual_seed(self.generator.initial_seed())
            self.pools = {}
            for name, count in self.counts.items():
                limit = self.seed_limits[name]
                if limit is None or not 1 <= limit <= count:
                    raise ValueError('Each subset corpus needs a valid explicit seed-pool size')
                self.pools[name] = torch.randperm(count, generator=pool_generator)[:limit]
        self.orders = {name: values[torch.randperm(len(values), generator=self.generator)] for name, values in self.pools.items()}
        self.cursors = {name: 0 for name in self.counts}
        self.next_graph = 0

    def state_dict(self):
        state = super().state_dict()
        state['fixed_seed_pools'] = {name: values.clone() for name, values in self.pools.items()}
        state['coverage_scope'] = 'configured_subset_rounds_not_full_graph_epochs'
        return state

    def load_state_dict(self, state):
        for name, values in self.pools.items():
            if not torch.equal(values, state['fixed_seed_pools'][name]):
                raise ValueError('Subset seed pool changed on resume')
        super().load_state_dict(state)


class SubsetLogFilter:
    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = record.msg.replace('Completed full declared epoch', 'Completed configured subset round')
            record.msg = record.msg.replace('step=%d epoch=%d', 'step=%d subset_round=%d')
        return True


class SubsetExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        global CURRENT_EXPERIMENT
        super().__init__(*args, **kwargs)
        CURRENT_EXPERIMENT = self
        self.logger.addFilter(SubsetLogFilter())
        self.logger.warning('SUBSET/OFFLINE protocol: eight configured subset rounds; pool sizes are disclosed reconstruction choices, not author-provided counts.')

    def metric(self, values):
        values = dict(values)
        if 'epoch' in values:
            values['subset_round'] = values.pop('epoch')
        super().metric(values)

    def status(self, value, **details):
        if 'completed_epochs' in details:
            details['completed_subset_rounds'] = details.pop('completed_epochs')
            details['completed_full_graph_epochs'] = 0
        if value == 'completed_pretraining':
            value = 'completed_subset_pretraining'
        super().status(value, **details)


class OfflinePPRSampler(PackedPPRSampler):
    def __init__(self, data, config):
        super().__init__(data, config)
        self.strict_offline = False
        if CURRENT_SCHEDULE is None:
            raise RuntimeError('The fixed subset schedule must exist before offline preparation')
        seeds = CURRENT_SCHEDULE.pools[data.audit['dataset']]
        started, cached = time.perf_counter(), 0
        logger = CURRENT_EXPERIMENT.logger if CURRENT_EXPERIMENT else None
        for index, seed in enumerate(seeds.tolist(), 1):
            if pause_requested():
                self.close()
                raise TrainingPaused('Offline PPR committed; cached neighborhoods will be reused on resume')
            if self.connection.execute('SELECT 1 FROM ppr WHERE seed=?', (seed,)).fetchone():
                cached += 1
            super().neighbors(seed)
            if logger and (index == 1 or index % 512 == 0 or index == len(seeds)):
                logger.info('OFFLINE_PPR dataset=%s prepared=%d/%d already_cached=%d elapsed=%.1fs; CUDA updates have not begun',
                            data.audit['dataset'], index, len(seeds), cached, time.perf_counter() - started)
        self.flush()
        self.strict_offline = True
        if CURRENT_EXPERIMENT:
            path = CURRENT_EXPERIMENT.directory / 'sampling' / (data.audit['dataset'] + '_offline_ppr.json')
            write_json(path, {'dataset': data.audit['dataset'], 'seeds': len(seeds), 'already_cached': cached,
                             'elapsed_seconds': time.perf_counter() - started, 'algorithm': self.VERSION,
                             'topology': data.audit['view_files_sha256']['edge_index.npy'],
                             'scope': 'All chosen PPR neighbor IDs precomputed; union-induced topology is still assembled per CUDA batch.'})

    def neighbors(self, seed):
        if self.strict_offline and not self.connection.execute('SELECT 1 FROM ppr WHERE seed=?', (seed,)).fetchone():
            raise RuntimeError('Offline PPR missing for seed=' + str(seed) + '; online expansion is forbidden in training')
        return super().neighbors(seed)


def write_subset_json(path, value):
    if Path(path).name == 'summary.json' and 'seed_coverage_current_epoch' in value:
        value = dict(value)
        value['completed_subset_rounds'] = value['completed_epochs']
        value['completed_epochs'] = 0
        value['complete_declared_subset_pretraining'] = value['complete_declared_pretraining']
        value['complete_declared_pretraining'] = False
        value['coverage_scope'] = 'fixed_seed_subset_rounds; full_graph_epochs_not_completed'
        value['seed_coverage_current_subset_round'] = value.pop('seed_coverage_current_epoch')
    write_json(path, value)


def save_subset_checkpoint(path, state):
    state = dict(state)
    rounds = state.get('completed_epochs', state.get('schedule', {}).get('epoch', 0))
    state['completed_subset_rounds'] = rounds
    state['completed_epochs'] = 0
    state['coverage_scope'] = 'fixed_seed_subset_rounds_not_full_graph_epochs'
    atomic_checkpoint(path, state)


def main():
    global CURRENT_EXPERIMENT, CURRENT_SCHEDULE
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/subset_offline_24h.json')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    if args.resume and args.warm_start:
        raise ValueError('Use either exact subset resume or explicit model-only warm start')
    original_model_factory = joint_train.make_model
    def make_subset_model(config):
        model = original_model_factory(config)
        if args.warm_start:
            state = torch.load(args.warm_start, map_location='cpu', weights_only=False)
            if state['config']['datasets'] != config['datasets'] or state['config']['model'] != config['model']:
                raise ValueError('Warm start changes model/corpus')
            model.load_state_dict(state['model'])
            write_json(CURRENT_EXPERIMENT.directory / 'warm_start.json', {
                'checkpoint': str(args.warm_start.resolve()), 'sha256': fingerprint(args.warm_start),
                'prior_optimizer_updates': state.get('step'), 'policy': 'Model weights transferred; Adam and new subset RNG restarted explicitly.'})
            del state
            gc.collect()
        return model
    previous_args = sys.argv
    previous = {name: getattr(joint_train, name) for name in ('Experiment', 'JointSchedule', 'PackedPPRSampler', 'write_json', 'atomic_checkpoint', 'make_model')}
    joint_train.Experiment = SubsetExperiment
    joint_train.JointSchedule = FixedSubsetSchedule
    joint_train.PackedPPRSampler = OfflinePPRSampler
    joint_train.write_json = write_subset_json
    joint_train.atomic_checkpoint = save_subset_checkpoint
    joint_train.make_model = make_subset_model
    sys.argv = [previous_args[0], '--config', str(args.config)]
    if args.resume:
        sys.argv += ['--resume', str(args.resume)]
    if args.max_steps is not None:
        sys.argv += ['--max-steps', str(args.max_steps)]
    try:
        joint_train.main()
    finally:
        sys.argv = previous_args
        for name, value in previous.items():
            setattr(joint_train, name, value)
        CURRENT_EXPERIMENT = None
        CURRENT_SCHEDULE = None


if __name__ == '__main__':
    main()
