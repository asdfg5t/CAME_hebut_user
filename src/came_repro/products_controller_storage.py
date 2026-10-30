"""Controller disk reservation without importing Torch/DGL a second time."""
import json
import os
import subprocess
import sys
from pathlib import Path

from .author_loader import ROOT
from .atomic_io import atomic_write_json


def reserve(report):
    from .storage_budget import size_bytes
    settings=json.loads((ROOT/'configs/storage.json').read_text(encoding='utf-8'))
    root=Path(settings['large_checkpoint_root'])
    owned=sum(size_bytes(x) for x in root.iterdir() if x.is_dir() and '_products_fas_' in x.name)
    atomic_write_json(ROOT/'artifacts/products_sampled_stream/storage_reservation.json',dict(root=str(root.resolve()),
        baseline_cache=report['external_cache_bytes'],baseline_owned=owned,reserved=report['requested_cache_bytes'],
        cap=int(settings['external_cache_high_water_gib']*2**30),minimum_free=int(settings['reserve_free_gib']*2**30)))


def _external(mode,project=0,cache=0):
    child=subprocess.run([sys.executable,'-m','came_repro.products_controller_storage','--mode',mode,
        '--project',str(project),'--cache',str(cache)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',
        env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    if child.returncode:raise RuntimeError('Products storage preflight failed: '+child.stderr[-5000:])
    return json.loads(child.stdout)


def external_budget_report(additional_project_bytes=0,additional_cache_bytes=0):
    return _external('report',additional_project_bytes,additional_cache_bytes)


def reserve_external(project,cache):return _external('reserve',project,cache)


if __name__=='__main__':
    import argparse
    from .storage_budget import check_budget
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['report','reserve'],required=True)
    parser.add_argument('--project',type=int,default=0);parser.add_argument('--cache',type=int,default=0)
    args=parser.parse_args();report=check_budget(additional_project_bytes=args.project,additional_cache_bytes=args.cache)
    if args.mode=='reserve':reserve(report)
    print(json.dumps(report),flush=True)
