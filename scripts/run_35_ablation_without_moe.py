"""PyCharm independent paper group: without_moe; default full seeds, --pilot diagnostic."""
import sys
from pathlib import Path
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from came_repro.paper_task_groups import run_group
if __name__ == '__main__':
    run_group('without_moe')

