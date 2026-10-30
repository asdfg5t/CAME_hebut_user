import copy
import hashlib
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import remaining_delivery as delivery


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def test_selection_uses_full_validation_and_ignores_test_score(tmp_path,monkeypatch):
    monkeypatch.setattr(delivery,'ROOT',tmp_path)
    candidates=[]
    for name,value,test,state,seed in [('a',.80,.99,'completed',0),('b',.81,.10,'completed',0),('partial',.95,1.,'running',0),('wrong_seed',.96,1.,'completed',1)]:
        directory=tmp_path/name
        write(directory/'summary.json',dict(seed=seed,validation_accuracy=value,test_accuracy=test,test_evaluations=0))
        write(directory/'status.json',dict(status=state));candidates.append(dict(validation_run=name,training_kind='fixed'))
    item,summary=delivery.best_validation(candidates,'wn')
    assert item['validation_run']=='b' and summary['validation_accuracy']==.81


def test_selection_cannot_change_inputs_after_freezing(tmp_path,monkeypatch):
    out=tmp_path/'output';release=tmp_path/'release.json';reference=tmp_path/'source_summary.json'
    write(release,dict(version=1));write(reference,dict(validation=.8))
    monkeypatch.setattr(delivery,'OUT',out);monkeypatch.setattr(delivery,'RELEASE',release)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    value=dict(release_sha256=sha(release),input_sha256={str(reference):sha(reference)},tasks={})
    write(out/'protocol_selection.json',value)
    assert delivery.freeze_selection({})[1]==value
    write(reference,dict(validation=.9))
    with pytest.raises(ValueError,match='输入改变'):delivery.freeze_selection({})


def test_delivery_refuses_shortened_data_and_seed_filtering():
    p=json.loads((ROOT/'configs/remaining_tasks_delivery_20261018.json').read_text())
    delivery.validate_profile(p)
    for change in [('seeds',[0,1]),('nodes',2000),('width',512),('official_test_nodes',100),('minimum_nodes_per_second',1)]:
        candidate=copy.deepcopy(p)
        if change[0]=='seeds':candidate['seeds']=change[1]
        else:candidate['capacity'][change[0]]=change[1]
        with pytest.raises(ValueError):delivery.validate_profile(candidate)


def test_partial_test_cannot_be_reported_as_complete(tmp_path):
    write(tmp_path/'summary.json',dict(test_evaluations=0))
    with pytest.raises(ValueError,match='完整官方测试'):delivery.audit_test('wn',tmp_path)
