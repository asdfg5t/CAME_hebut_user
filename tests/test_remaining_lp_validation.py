import numpy as np
import pytest
from came_repro.remaining_lp_validation import verify_record
from came_repro.author_loader import fingerprint


def fixture_record(tmp_path):
    path = tmp_path/'test_link_ranks.npz'
    rank = np.array([1.,2.5,151.], dtype=np.float32)
    np.savez_compressed(path,rank=rank)
    return {'run_directory':str(tmp_path),'artifact_sha256':{str(path):fingerprint(path)},
            'summary':{'test_evaluations':1,'test':{'queries':3,'official':{'mrr':float(np.mean(1/rank.astype(np.float64)))}}}}


def test_saved_tied_ranks_match_summary_and_reject_wrong_mrr(tmp_path):
    record = fixture_record(tmp_path)
    verify_record(record)
    record['summary']['test']['official']['mrr'] = .9
    with pytest.raises(ValueError):verify_record(record)


def test_changed_result_file_cannot_be_reused(tmp_path):
    record = fixture_record(tmp_path)
    np.savez_compressed(tmp_path/'test_link_ranks.npz',rank=np.array([1.,1.,1.]))
    with pytest.raises(ValueError):verify_record(record)
