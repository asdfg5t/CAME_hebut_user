import logging
from pathlib import Path
import sys
import time

from came_repro import budgeted_pipeline as budget
from came_repro.ide_workflow import read_json, write_json, CORPORA


def profile(hours=24):
    return read_json(budget.ROOT / 'configs' / ('budgeted_' + str(hours) + 'h.json'))


def test_budget_preserves_full_domains_in_24h_and_declares_12h_omissions():
    full, representative = profile(24), profile(12)
    budget.validate_budget(full)
    budget.validate_budget(representative)
    assert len(budget.task_plan(full)) == 13
    assert len(budget.task_plan(representative)) == 7
    assert ('node', 'ogbn-products', 'representation') in budget.task_plan(full)
    assert ('node', 'ogbn-products', 'representation') not in budget.task_plan(representative)
    for config in (full, representative):
        assert config['evaluation_seed'] == 0 and config['wiki_split_id'] == 0
        for protocol in ('representation', 'few_shot'):
            original = read_json(budget.ROOT / 'configs' / ('joint_' + protocol + '_reconstruction.json'))
            assert original['model']['hidden_dim'] == 1024
            assert original['model']['experts_per_view'] == 6
            assert original['model']['w_tv'] == .5
            assert original['training']['epochs'] == 8 and original['training']['max_steps'] is None


def test_short_head_budgets_keep_official_task_and_cuda_module_entry():
    args = budget.evaluation_arguments(profile(), 'node', 'wiki-cs', 'came', Path('partial_checkpoint.pt'))
    assert args[:2] == ['-m', 'came_repro.downstream']
    assert args[args.index('--checkpoint') + 1] == 'partial_checkpoint.pt'
    assert args[args.index('--epochs') + 1] == '300'
    assert '--probe-batch-size' not in args  # no silent minibatch optimizer adaptation
    lp = budget.evaluation_arguments(profile(), 'link', 'books-lp', 'clip_text')
    assert lp[:2] == ['-m', 'came_repro.link_evaluation']
    assert lp[lp.index('--epochs') + 1] == '20'


def test_checkpoint_reuse_chooses_consistent_last_not_probe_or_ablation(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, 'ROOT', tmp_path)
    def run(name, variant='came', evaluation=False):
        path = tmp_path / 'runs' / name
        (path / 'checkpoints').mkdir(parents=True)
        config = {'datasets': CORPORA['representation'], 'seed': 42, 'variant': variant}
        if evaluation:
            config['evaluation'] = {'task': 'node'}
        write_json(path / 'config.json', config)
        write_json(path / 'manifest.json', {'checkpoint_directory': str(path / 'checkpoints')})
        (path / 'checkpoints/last.pt').write_bytes(b'selection_fixture')
        return path
    selected = run('proper')
    run('ablation', 'without_dimension_gate')
    run('probe', evaluation=True)
    result = budget.newest_checkpoint('representation', 42)
    assert result[2] == selected
    assert budget.newest_checkpoint('representation', 42, before={'proper', 'ablation', 'probe'}) is None


def test_watchdog_times_out_silent_child_without_starting_gpu_work():
    started = time.monotonic()
    outcome = budget.run_bounded(['-c', 'import time; time.sleep(10)'], .15, logging.getLogger('watchdog_fixture'))
    assert outcome['timed_out']
    assert outcome['returncode'] != 0
    assert time.monotonic() - started < 5
    assert outcome['run_directory'] is None


def test_wall_stop_cannot_be_reported_as_successful_training():
    row = budget.result_row({'task': 'pretraining'}, {'timed_out': True, 'returncode': 0, 'run_directory': None})
    assert row['status'] == 'time_budget_reached'
