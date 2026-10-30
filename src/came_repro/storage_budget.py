"""Allocation guards for the explicitly configured project and experiment cache."""
import json
import shutil
from pathlib import Path

from .author_loader import ROOT


def size_bytes(directory):
    seen, total = set(), 0
    for path in Path(directory).rglob('*'):
        if not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += stat.st_size
    return total


def check_budget(additional_project_bytes=0, additional_cache_bytes=0):
    if additional_project_bytes < 0 or additional_cache_bytes < 0:
        raise ValueError('Additional storage reservations cannot be negative')
    config = json.loads((ROOT / 'configs/storage.json').read_text())
    roots = [Path(config[key]) for key in ('large_checkpoint_root', 'embedding_cache_root')]
    project = size_bytes(ROOT)
    cache = sum(size_bytes(path) for path in roots)
    project_free = shutil.disk_usage(ROOT).free
    cache_free = shutil.disk_usage(roots[0].parent).free
    reserve = config['reserve_free_gib'] * 2**30
    valid = (project + additional_project_bytes <= config['project_high_water_gib'] * 2**30 and
             cache + additional_cache_bytes <= config['external_cache_high_water_gib'] * 2**30 and
             project_free - additional_project_bytes >= reserve and
             cache_free - additional_cache_bytes >= reserve)
    report = {'project_unique_file_bytes': project, 'external_cache_bytes': cache,
              'project_free_bytes': project_free, 'cache_free_bytes': cache_free,
              'project_high_water_gib': config['project_high_water_gib'],
              'cache_high_water_gib': config['external_cache_high_water_gib'],
              'reserve_free_gib': config['reserve_free_gib'],
              'requested_project_bytes': additional_project_bytes,
              'requested_cache_bytes': additional_cache_bytes, 'within_budget': valid,
              'raw_and_experiment_records_deleted': False}
    if not valid:
        raise RuntimeError('Storage budget exceeded. Preserve raw data/checkpoints/logs; remove only identified regenerable caches or change the explicit budget. ' + json.dumps(report))
    return report


def guard_allocation(path, additional_bytes):
    config = json.loads((ROOT / 'configs/storage.json').read_text())
    target = Path(path).resolve()
    if any(target.is_relative_to(Path(config[key]).resolve()) for key in ('large_checkpoint_root', 'embedding_cache_root')):
        return check_budget(additional_cache_bytes=additional_bytes)
    if target.is_relative_to(ROOT.resolve()):
        return check_budget(additional_project_bytes=additional_bytes)
    # Generic atomic-checkpoint utilities also serve temporary test fixtures.
    free = shutil.disk_usage(target.parent).free
    if free - additional_bytes < config['reserve_free_gib'] * 2**30:
        raise RuntimeError('Insufficient disk reserve for atomic checkpoint')
    return {'free_bytes': free, 'requested_bytes': additional_bytes}


def tensor_storage_bytes(value):
    """Unique tensor storages; include a margin for serialization and scalar metadata."""
    import torch
    seen = set()
    total = 0
    def visit(item):
        nonlocal total
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
    visit(value)
    return int(total * 1.05) + 1024**2
