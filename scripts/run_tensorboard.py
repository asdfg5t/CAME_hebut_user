"""PyCharm: view existing run scalars on localhost:6006."""
import sys
from pathlib import Path
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from came_repro.ide_workflow import main
if __name__ == '__main__':
    main('tensorboard')
