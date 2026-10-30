"""Run alongside the suite: request pause at a coherent optimizer/epoch boundary."""
import json
from pathlib import Path
import datetime
import sys
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from came_repro.atomic_io import atomic_write_json


def main():
    root = Path(__file__).resolve().parents[1]
    legacy = '--legacy-full-suite' in sys.argv
    pointer = root / ('artifacts/workflow_state/full_suite_current.json' if legacy else 'artifacts/workflow_state/corrected_node_suite_current.json')
    if not pointer.is_file():
        raise RuntimeError('No current ledger; start scripts/run_full_suite.py first')
    path = Path(json.loads(pointer.read_text(encoding='utf-8'))['pause_file']).resolve()
    if not path.is_relative_to(root / 'artifacts/workflow_state'):
        raise ValueError('Pause request path must stay inside this project workflow state')
    atomic_write_json(path, {'reason':'user_safe_pause','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print('Pause requested. Wait for SAFE_PAUSE and controller state=paused; run the same entry to resume into a new run.')


if __name__ == '__main__':
    main()
