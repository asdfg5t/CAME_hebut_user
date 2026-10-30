"""Pause the quality pilot at a coherent checkpoint or inference cursor."""
import sys
from pathlib import Path
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from came_repro.author_loader import ROOT
from came_repro.atomic_io import read_json_file, atomic_write_json
if __name__ == '__main__':
    pointer = read_json_file(ROOT/'artifacts/workflow_state/quality_pilot_current.json')
    path = Path(pointer['pause_file']).resolve()
    if not path.is_relative_to((ROOT/'artifacts/workflow_state').resolve()):
        raise ValueError('Pause path outside this project')
    atomic_write_json(path,{'reason':'user_quality_safe_pause'})
    print('Quality pause requested; wait for SAFE_PAUSE / QUALITY_STATE=paused.')
