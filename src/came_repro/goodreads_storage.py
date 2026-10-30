"""Bounded checkpoint reservation after a full stage budget audit.

Only the Goodreads workflow uses this writer. It avoids scanning the entire
venv before every tiny checkpoint; it preserves atomic/fsync persistence and
checks the reserved family allocation and live free space on every write.
"""
import json
import shutil
from pathlib import Path

from .author_loader import ROOT
from .atomic_io import atomic_write_json,unique_temporary_path,atomic_replace,retry_windows_io
from .filesystem_paths import windows_io_path

RESERVATION=ROOT/'artifacts/goodreads_official_rebuild/storage_reservation.json'


def owned_checkpoint_bytes(checkpoint_root):
    total=0
    for run in Path(checkpoint_root).iterdir():
        if run.is_dir() and '_goodreads_' in run.name:
            total+=sum(p.stat().st_size for p in run.iterdir() if p.is_file())
    return total


def reserve(report):
    config=json.loads((ROOT/'configs/storage.json').read_text(encoding='utf-8'))
    if not report['within_budget'] or report['requested_cache_bytes']<2**30:raise ValueError('Full audited1GiB reservation required')
    checkpoint_root=Path(config['large_checkpoint_root']).resolve()
    value=dict(baseline_external_cache_bytes=report['external_cache_bytes'],baseline_owned_bytes=owned_checkpoint_bytes(checkpoint_root),
        max_extra_cache_bytes=report['requested_cache_bytes'],cache_cap_bytes=int(config['external_cache_high_water_gib']*2**30),
        reserve_free_bytes=int(config['reserve_free_gib']*2**30),checkpoint_root=str(checkpoint_root),
        full_stage_budget_audit=report,scope='Only this serial Goodreads checkpoint family; stage boundaries run the full original budget audit')
    atomic_write_json(RESERVATION,value)


def guard(path,additional_bytes):
    value=json.loads(RESERVATION.read_text(encoding='utf-8'))
    target=Path(path).resolve();root=Path(value['checkpoint_root']).resolve()
    if not target.is_relative_to(root) or len(target.relative_to(root).parts)!=2 or '_goodreads_' not in target.parent.name:
        raise ValueError('Checkpoint escaped the reserved Goodreads family')
    if additional_bytes<0:raise ValueError('Invalid allocation estimate')
    owned=owned_checkpoint_bytes(root)
    if max(0,owned-value['baseline_owned_bytes'])+additional_bytes>value['max_extra_cache_bytes']:
        raise RuntimeError('Goodreads reserved checkpoint allocation exhausted')
    if value['baseline_external_cache_bytes']-value['baseline_owned_bytes']+owned+additional_bytes>value['cache_cap_bytes']:
        raise RuntimeError('External cache cap would be exceeded')
    if shutil.disk_usage(root).free-additional_bytes<value['reserve_free_bytes'] or shutil.disk_usage(ROOT).free<value['reserve_free_bytes']:
        raise RuntimeError('Live free-space reserve would be exceeded')


def atomic_checkpoint(path,state):
    import os
    import numpy as np
    import torch
    from .storage_budget import tensor_storage_bytes
    seen=set()
    def numpy_bytes(item):
        if isinstance(item,np.ndarray):
            if id(item) in seen:return 0
            seen.add(id(item));return item.nbytes
        if isinstance(item,dict):return sum(numpy_bytes(v) for v in item.values())
        if isinstance(item,(tuple,list)):return sum(numpy_bytes(v) for v in item)
        return 0
    estimate=tensor_storage_bytes(state)+numpy_bytes(state)+2**20
    path=Path(path);guard(path,estimate);temporary=unique_temporary_path(path)
    def save():
        with open(windows_io_path(temporary),'wb') as stream:
            torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    retry_windows_io(save,temporary)
    guard(temporary,0)
    atomic_replace(temporary,path)
