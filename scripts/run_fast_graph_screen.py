"""PyCharm: correct graph protocols, reuse encoder, validate then confirm seed zero."""
import sys
from pathlib import Path
if __name__=='__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from came_repro.fast_graph_screen import main
if __name__=='__main__':
    raise SystemExit(main())
