"""Pause the stability check at a committed inference/optimizer boundary."""
import sys
from pathlib import Path
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from came_repro.author_loader import ROOT
from came_repro.atomic_io import read_json_file, atomic_write_json
if __name__ == '__main__':
    pointer=read_json_file(ROOT/'artifacts/workflow_state/corrected_node_stability_current.json')
    pause=Path(pointer['pause_file']).resolve()
    if not pause.is_relative_to((ROOT/'artifacts/workflow_state').resolve()):
        raise ValueError('Pause path escaped workflow state')
    atomic_write_json(pause,{'reason':'user_safe_pause'})
    print('Pause requested; wait for SAFE_PAUSE and CORRECTED_STATE=paused.')
