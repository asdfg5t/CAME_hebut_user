"""The basic Python launcher must transfer arguments and fail safely before imports."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_environment', ROOT / 'scripts/_run_environment.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def entry_and_environment(tmp_path):
    entry = tmp_path / 'project/scripts/run_00_main_pipeline.py'
    entry.parent.mkdir(parents=True)
    entry.write_text('# test entry')
    expected = entry.parents[1] / '.venv-came39/Scripts/python.exe'
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b'test interpreter path')
    return entry, expected


def test_bridge_preserves_arguments_cwd_and_child_failure(tmp_path, monkeypatch):
    entry, expected = entry_and_environment(tmp_path)
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'basic/python.exe'))
    monkeypatch.setattr(sys, 'argv', [str(entry), '--workflow-config', 'settings with spaces.json', '--dry-run'])
    monkeypatch.chdir(tmp_path)
    calls = []
    class Child:
        def wait(self):
            return 7
    def launch(command, **options):
        calls.append((command, options))
        return Child()
    monkeypatch.setattr(bridge.subprocess, 'Popen', launch)
    with pytest.raises(SystemExit) as outcome:
        bridge.ensure_project_environment(entry)
    assert outcome.value.code == 7
    command, options = calls[0]
    assert command == [str(expected), '-u', str(entry), '--workflow-config', 'settings with spaces.json', '--dry-run']
    assert options['cwd'] == str(tmp_path)
    assert options['env']['PYTHONUNBUFFERED'] == '1'
    assert len(calls) == 1


def test_bridge_missing_environment_never_runs_in_basic_python(tmp_path, monkeypatch):
    entry = tmp_path / 'project/scripts/run_00_main_pipeline.py'
    entry.parent.mkdir(parents=True)
    entry.write_text('# test entry')
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'basic/python.exe'))
    monkeypatch.setattr(bridge.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('No subprocess should start'))
    with pytest.raises(RuntimeError, match='environment is missing'):
        bridge.ensure_project_environment(entry)


def test_bridge_already_in_project_environment_does_not_recurse(tmp_path, monkeypatch):
    entry, expected = entry_and_environment(tmp_path)
    monkeypatch.setattr(sys, 'executable', str(expected))
    monkeypatch.setattr(bridge.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('No recursive subprocess'))
    assert bridge.ensure_project_environment(entry) is None
