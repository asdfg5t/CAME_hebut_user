import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/run_validated_remaining_tasks.py'
spec=importlib.util.spec_from_file_location('validated_remaining_entry',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_unverified_or_wrong_protocol_task_cannot_enter_ready_file():
    p={'encoder_sha256':'frozen','paper_accuracy_pct':94.59,'preferred_absolute_gap_pp':2}
    good={'dataset':'FB15K237','seed':0,'test_evaluations':1,'graph_policy':'train_reverse','feature_transform':'raw','encoder_sha256':'frozen','test_accuracy':.9348187}
    assert module.verified_seed0(p,good)[1]==pytest.approx(-1.10813)
    for wrong in [{'test_evaluations':0},{'test_accuracy':.89},{'dataset':'WN18RR'},{'feature_transform':'train_pair_affine'},{'seed':1}]:
        with pytest.raises(ValueError):module.verified_seed0(p,dict(good,**wrong))
