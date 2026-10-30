import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('aggregate_results',ROOT/'scripts/aggregate_results.py')
aggregate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


def make_run(path,seed,accuracy,complete=True):
    path.mkdir()
    contents={'summary.json':{'dataset':'wiki-cs','task':'node','representation':'came','pretraining_complete':complete,
                              'test_accuracy':accuracy,'max_epochs':2000},
              'config.json':{'seed':seed,'evaluation':{'epochs':2000,'patience':100,'split_id':0}},
              'manifest.json':{'harness_sources':{'src/came_repro/downstream.py':'fixture_v1'},
                               'data':{'prepared_manifest_sha256':'fixture_data'}},
              'status.json':{'status':'completed_downstream_task'}}
    for filename,content in contents.items():
        (path/filename).write_text(json.dumps(content))
    return path


def test_aggregate_preserves_seed_numbers_and_sample_std(tmp_path,monkeypatch):
    runs=[make_run(tmp_path/'seed0',0,.6),make_run(tmp_path/'seed1',1,.8)]
    output=tmp_path/'aggregate'
    monkeypatch.setattr(sys,'argv',['aggregate_results','--runs',*map(str,runs),'--expected-seeds','2','--output',str(output)])
    aggregate.main()
    report=json.loads((output/'aggregate.json').read_text())
    metric=report['per_split']['0']['test_accuracy']
    assert metric['mean']==pytest.approx(.7)
    assert metric['sample_std']==pytest.approx(np.std([.6,.8],ddof=1))
    assert report['complete_requested_grid']
    assert not report['paper_reproduction_complete']
    assert len((output/'seed_metrics.csv').read_text().splitlines())==3


def test_aggregate_rejects_partial_and_duplicate_seed(tmp_path,monkeypatch):
    partial=make_run(tmp_path/'partial',0,.6,False)
    monkeypatch.setattr(sys,'argv',['aggregate_results','--runs',str(partial),'--output',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='Incomplete pretraining'):
        aggregate.main()
    duplicate=make_run(tmp_path/'duplicate',0,.8)
    monkeypatch.setattr(sys,'argv',['aggregate_results','--runs',str(partial),str(duplicate),'--allow-partial','--output',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='Duplicate seed'):
        aggregate.main()
