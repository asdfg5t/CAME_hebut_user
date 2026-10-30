import json
from pathlib import Path
import numpy as np
from came_repro.author_loader import ROOT
from came_repro.fast_graph_screen import choose_policy, validate_profile, wiki_confirmation


def test_fast_scope_reuses_encoder_and_excludes_ele():
    profile=json.loads((ROOT/'configs/fast_graph_screen.json').read_text())
    validate_profile(profile)
    assert profile['extra_pretraining_updates']==0
    assert 'ele-fashion' not in profile['datasets']
    assert profile['cumulative_time_limit'] is None
    assert choose_policy(.70,.799,.5)=='bidirected'
    assert choose_policy(.70,.704,.5)=='original'


def test_reused_wiki_test_matches_complete_official_saved_predictions():
    lock=json.loads((ROOT/'artifacts/audit/wiki_fast_confirmation_lock.json').read_text())
    record=wiki_confirmation(lock['encoder_sha256'])
    assert record['summary']['test_evaluations']==1
    with np.load(Path(record['run_directory'])/'test_predictions.npz') as output:
        accuracy=float(np.mean(output['prediction']==output['labels']))
        labels=np.load(ROOT/'data/processed/text_graphs/wiki-cs/labels.npy',mmap_mode='r')
        np.testing.assert_array_equal(output['labels'],labels[output['node_ids']])
        mask=np.load(ROOT/'data/processed/text_graphs/wiki-cs/test_mask_indices.npy',mmap_mode='r')
        np.testing.assert_array_equal(output['node_ids'],np.flatnonzero(mask))
    assert accuracy==record['summary']['test_accuracy']
