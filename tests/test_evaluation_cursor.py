import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from came_repro.evaluation_cursor import EvaluationCursor


def test_resume_keeps_committed_rows_and_requires_same_head_and_order(tmp_path):
    first=tmp_path/'one';second=tmp_path/'two';first.mkdir();second.mkdir()
    identity=dict(head='shaA',data='official',batch=64)
    cursor=EvaluationCursor(first/'cursor.json',identity,5,'float64');cursor.commit(0,[1.,2.5])
    with pytest.raises(ValueError):cursor.commit(1,[3])
    with pytest.raises(ValueError):cursor.complete_values()
    resumed=EvaluationCursor(second/'cursor.json',identity,5,'float64',first/'cursor.json')
    assert resumed.cursor==2
    resumed.commit(2,[3.,4.,5.]);np.testing.assert_array_equal(resumed.complete_values(),[1,2.5,3,4,5])
    with pytest.raises(ValueError,match='identity'):EvaluationCursor(second/'other.json',dict(identity,head='shaB'),5,'float64',second/'cursor.json')


def test_resume_rejects_modified_prefix_before_scoring(tmp_path):
    cursor=EvaluationCursor(tmp_path/'cursor.json',dict(head='frozen'),3,'int64');cursor.commit(0,[1,2])
    values=np.load(cursor.values_path,mmap_mode='r+');values[0]=99;values.flush();values._mmap.close()
    with pytest.raises(ValueError,match='prefix'):EvaluationCursor(tmp_path/'new.json',dict(head='frozen'),3,'int64',tmp_path/'cursor.json')
