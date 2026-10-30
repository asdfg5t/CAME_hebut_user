"""Pause the graph screen at a saved head boundary or inference cursor."""
import sys
from pathlib import Path
if __name__=='__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from came_repro.author_loader import ROOT
from came_repro.atomic_io import read_json_file,atomic_write_json
if __name__=='__main__':
    p=read_json_file(ROOT/'artifacts/workflow_state/fast_graph_screen_current.json')
    path=Path(p['pause_file']).resolve()
    if not path.is_relative_to((ROOT/'artifacts/workflow_state').resolve()):raise ValueError('Invalid pause path')
    atomic_write_json(path,{'reason':'user_fast_graph_pause'})
    print('Pause requested; wait for SAFE_PAUSE / FAST_STATE=paused.')
