import copy
import json
from pathlib import Path

import numpy as np
import pytest

from came_repro import corrected_node_suite as suite


def profile():
    return json.loads((suite.ROOT/'configs/corrected_four_node_tasks.json').read_text())


def test_repeat_grid_has_complete_paper_membership_and_preserves_ele():
    jobs = suite.build_jobs(profile())
    wiki = [j for j in jobs if j['dataset'] == 'wiki-cs']
    assert len(jobs) == 430
    assert {(j['split_id'], j['seed']) for j in wiki} == {(s,r) for s in range(20) for r in range(20)}
    for dataset in ('ele-fashion', 'ogbn-arxiv', 'books-nc'):
        assert {j['seed'] for j in jobs if j['dataset']==dataset} == set(range(10))
    with pytest.raises(ValueError, match='Ele-fashion'):
        suite.child_arguments(jobs[0], profile(), Path('encoder.pt'), {})


def test_predeclared_stability_grid_is_partial_not_full_paper():
    jobs = suite.build_jobs(profile(), stability_check=True)
    assert len(jobs)==19
    assert {(j['seed'],j['split_id']) for j in jobs if j['dataset']=='wiki-cs'} == {(0,0),(1,0),(2,0)}
    assert len(suite.build_jobs(profile(), single_seed=True))==13


def test_protocol_change_or_seed_dropping_rejected():
    for key, value in [('graph_policies', dict.fromkeys(profile()['datasets'], 'original')),
                       ('probe_epochs', 1), ('wiki_seeds', [0]), ('extra_pretraining_updates', 1)]:
        altered = profile(); altered[key] = value
        with pytest.raises(ValueError):
            suite.build_jobs(altered)


def test_confirmation_rejects_missing_official_test_node_and_invented_accuracy(tmp_path):
    directory = tmp_path/'data/processed/mm_graph/books-nc'; directory.mkdir(parents=True)
    np.save(directory/'labels.npy', np.array([0,1,0,1]))
    np.save(directory/'test_indices.npy', np.array([1,2,3]))
    run = tmp_path/'run'; run.mkdir()
    summary = {'dataset':'books-nc','seed':1,'split_id':0,'graph_policy':'bidirected',
               'test_evaluations':1,'test_accuracy':1.,'max_epochs':2000,'patience':100,
               'lr':.05,'optimizer':'Adam','training_batch_size':0}
    (run/'summary.json').write_text(json.dumps(summary))
    (run/'status.json').write_text(json.dumps({'status':'completed_quality_probe'}))
    job = {'dataset':'books-nc','seed':1,'split_id':0,'policy':'bidirected'}
    record = {'state':'completed','run_directory':str(run),'summary':summary,'artifact_sha256':{}}
    np.savez(run/'test_predictions.npz',node_ids=[1,2],labels=[1,0],prediction=[1,0])
    with pytest.raises(AssertionError):
        suite.verify_record(record,job,root=tmp_path)
    np.savez(run/'test_predictions.npz',node_ids=[1,2,3],labels=[1,0,1],prediction=[1,0,0])
    with pytest.raises(ValueError, match='claimed score'):
        suite.verify_record(record,job,root=tmp_path)


def test_command_reuses_head_cursor_and_fixed_encoder(monkeypatch):
    monkeypatch.setattr(suite,'best_resume',lambda record, filename: Path('paused/probe_last.pt'))
    job={'dataset':'books-nc','seed':2,'split_id':0,'policy':'bidirected'}
    args=suite.child_arguments(job,profile(),Path('same_encoder.pt'),{})
    assert args[args.index('--checkpoint')+1]=='same_encoder.pt'
    assert args[args.index('--resume-head')+1]=='paused\\probe_last.pt' or args[args.index('--resume-head')+1]=='paused/probe_last.pt'
    assert args[args.index('--inference-batch-size')+1]=='128'
    assert '--confirm-test' in args


def test_imported_four_results_are_real_complete_official_tests():
    _, imported, provenance = suite.import_confirmations(profile(), suite.build_jobs(profile()))
    assert len(imported) >= 13
    assert sum(r['job']['dataset']=='ele-fashion' for r in imported.values())==10
    assert imported['nc_books-nc_0_0']['verification']['official_test_nodes']==203428
    assert imported['nc_wiki-cs_0_0']['verification']['official_test_nodes']==5847
    assert imported['nc_ogbn-arxiv_0_0']['verification']['official_test_nodes']==48603
    assert all(provenance['selections'][dataset]['locked_before_test'] for dataset in ('wiki-cs','ogbn-arxiv','books-nc'))
