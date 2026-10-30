import copy
from pathlib import Path
import pytest
from came_repro import subset_day_pipeline as day
from came_repro.ide_workflow import read_json, write_json


def test_quality_budget_keeps_full_model_eight_subset_rounds_and_full_probe_limit():
    plan = read_json(day.ROOT / 'configs/subset_day_pipeline.json')
    training = read_json(day.ROOT / plan['training_config'])
    day.validate_plan(plan, training)
    assert day.update_budget(training)['planned_batches'] == 10240
    assert not day.update_budget(training)['full_graph_epoch_claim']
    args = day.probe_arguments(plan, 2, 'came', Path('encoder.pt'))
    assert args[args.index('--epochs') + 1] == '2000'
    assert args[args.index('--seed') + 1] == '2'
    broken = copy.deepcopy(training)
    broken['training']['max_steps'] = 10
    with pytest.raises(ValueError, match='without a step cap'):
        day.validate_plan(plan, broken)


def test_aggregate_uses_successful_actual_seeds_and_sample_sd_only():
    rows = [{'task':'probe','variant':'came','status':'completed','summary':{'test_accuracy':value}} for value in (.8,.9)]
    rows += [{'task':'probe','variant':'came','status':'time_budget_reached','summary':{'test_accuracy':1.0}},
             {'task':'probe','variant':'clip_image','status':'deferred'}]
    result = day.aggregate_results(rows)
    assert len(result) == 1
    assert result[0]['completed_downstream_seeds'] == 2
    assert result[0]['mean_accuracy_percent'] == 85
    assert result[0]['sample_std_percent'] == pytest.approx(7.0710678118654755)


def test_checkpoint_selection_is_from_exact_child_run_not_other_historical_encoder(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    checkpoint = tmp_path / 'cache'
    checkpoint.mkdir()
    write_json(run / 'manifest.json', {'checkpoint_directory':str(checkpoint)})
    assert day.own_checkpoint(str(run)) is None
    (checkpoint / 'last.pt').write_bytes(b'consistent_partial_fixture')
    assert day.own_checkpoint(str(run)) == checkpoint / 'last.pt'
    (checkpoint / 'encoder_final.pt').write_bytes(b'complete_subset_fixture')
    assert day.own_checkpoint(str(run)) == checkpoint / 'encoder_final.pt'
    assert day.own_checkpoint(None) is None
