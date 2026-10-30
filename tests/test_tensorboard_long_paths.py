import os
from pathlib import Path
import pytest
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from came_repro.filesystem_paths import windows_io_path
from came_repro.experiment import compact_run_name


def test_long_experiment_names_remain_distinct_and_compact_for_source_snapshots():
    name='suite_pretrain_representation_without_vision_fusion_loss_seed42'
    first=compact_run_name('20261011T104157Z',name,42,'a221c2')
    second=compact_run_name('20261011T104157Z',name+'_other',42,'a221c2')
    assert first!=second and len(first)<=50
    assert first.startswith('20261011T104157Z_suite_pretra_')


@pytest.mark.skipif(os.name != 'nt', reason='Windows native long-path regression')
def test_long_windows_event_file_is_written_and_read_without_disabling_tensorboard(tmp_path):
    directory = tmp_path / ('nested_' + 'x' * 140) / 'tensorboard'
    path = windows_io_path(directory)
    assert path.startswith('\\\\?\\')
    Path(path).mkdir(parents=True, exist_ok=True)
    with SummaryWriter(path) as writer:
        writer.add_scalar('logger_verification/value', .125, 7)
        writer.flush()
    events = list(Path(path).glob('events.*'))
    assert events and len(str(events[0])) > 260
    accumulator = EventAccumulator(path).Reload()
    values = accumulator.Scalars('logger_verification/value')
    assert len(values) == 1 and values[0].step == 7
    assert values[0].value == .125
    assert windows_io_path(Path(path)) == path
