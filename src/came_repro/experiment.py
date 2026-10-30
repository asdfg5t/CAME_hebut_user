"""Unique runs, dual structured logging, source/data provenance, atomic checkpoints."""
import csv
import datetime
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import psutil
import torch
import dgl
from torch.utils.tensorboard import SummaryWriter
from .author_loader import ROOT, fingerprint, verify_originals
from .filesystem_paths import windows_io_path
from .atomic_io import atomic_write_json, atomic_replace, unique_temporary_path, retry_windows_io


def write_json(path, value):
    atomic_write_json(path, value)


def compact_run_name(stamp, name, seed, suffix):
    # Preserve the full semantic name in config.json; the digest disambiguates
    # labels with the same prefix while keeping source snapshots below MAX_PATH.
    digest = hashlib.sha256(name.encode('utf-8')).hexdigest()[:6]
    return f'{stamp}_{name[:12]}_{digest}_s{seed}_{suffix}'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # DGL reductions are not necessarily deterministic. Record this, do not promise bitwise replay.


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def atomic_checkpoint(path, state):
    from .storage_budget import guard_allocation, tensor_storage_bytes
    guard_allocation(path, tensor_storage_bytes(state))
    temporary = unique_temporary_path(path)
    def save():
        with open(windows_io_path(temporary), 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
    retry_windows_io(save, temporary)
    atomic_replace(temporary, path)


class Experiment:
    def __init__(self, config, data_audit, parent=None):
        self.config, self.data_audit = config, data_audit
        if config.get('storage'):
            from .storage_budget import check_budget
            check_budget()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.directory = ROOT / 'runs' / compact_run_name(stamp, config['name'], config['seed'], uuid.uuid4().hex[:6])
        self.directory.mkdir(parents=True, exist_ok=False)
        for name in ('logs', 'checkpoints', 'figures', 'tensorboard', 'source'):
            (self.directory / name).mkdir()
        checkpoint_root = config.get('storage', {}).get('large_checkpoint_root')
        self.checkpoint_directory = Path(checkpoint_root) / self.directory.name if checkpoint_root else self.directory / 'checkpoints'
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        if checkpoint_root:
            write_json(self.directory / 'checkpoints/location.json', {'checkpoint_directory': str(self.checkpoint_directory)})
        self.logger = logging.getLogger(str(self.directory))
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        formatter.converter = __import__('time').gmtime
        for handler in (logging.StreamHandler(), logging.FileHandler(self.directory / 'logs/train.log', encoding='utf-8')):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        # TensorBoard's stub turns filenames into bytes: on Windows even a 261
        # character event path fails without the native extended-path prefix.
        self.writer = SummaryWriter(windows_io_path(self.directory / 'tensorboard'))
        self.jsonl = (self.directory / 'logs/metrics.jsonl').open('w', encoding='utf-8')
        self.csvfile = (self.directory / 'logs/metrics.csv').open('w', newline='', encoding='utf-8')
        self.csvwriter = None
        write_json(self.directory / 'config.json', config)
        originals = verify_originals()
        sources = {}
        for pattern in ('src/came_repro/*.py', 'scripts/*.py', 'scripts/*.ps1', 'configs/**/*.json', 'pyproject.toml'):
            for path in ROOT.glob(pattern):
                relative = path.relative_to(ROOT)
                destination = self.directory / 'source' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                sources[str(relative)] = fingerprint(path)
        for name in originals:
            if name.endswith('.py'):
                shutil.copy2(ROOT / name, self.directory / 'source' / name)
        driver = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=15)
        manifest = {'created_utc': stamp, 'log_timezone': 'UTC', 'user_timezone': 'Asia/Shanghai',
                    'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                    'torch': torch.__version__, 'dgl': dgl.__version__, 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(0), 'driver': driver.stdout.strip(),
                    'ram_gib': psutil.virtual_memory().total / 2**30,
                    'original_files': originals, 'harness_sources': sources, 'data': data_audit,
                    'command': sys.argv, 'parent_checkpoint': str(parent) if parent else None,
                    'parent_sha256': fingerprint(parent) if parent else None,
                    'checkpoint_directory': str(self.checkpoint_directory),
                    'determinism': 'seeded RNG; TF32 off; DGL CUDA scatter reductions may be nondeterministic'}
        write_json(self.directory / 'manifest.json', manifest)
        freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (self.directory / 'source/requirements.freeze.txt').write_text(freeze, encoding='utf-8')
        self.status('running')
        print('RUN_DIR=' + str(self.directory), flush=True)

    def status(self, value, **details):
        write_json(self.directory / 'status.json', {'status': value, 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **details})

    def metric(self, values):
        self.jsonl.write(json.dumps(values, allow_nan=False) + '\n')
        self.jsonl.flush()
        if self.csvwriter is None:
            self.csvwriter = csv.DictWriter(self.csvfile, fieldnames=list(values))
            self.csvwriter.writeheader()
        self.csvwriter.writerow(values)
        self.csvfile.flush()
        for key, value in values.items():
            if isinstance(value, (float, int)):
                self.writer.add_scalar(key, value, values['step'])
        self.writer.flush()

    def close(self):
        self.jsonl.close()
        self.csvfile.close()
        self.writer.close()
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
