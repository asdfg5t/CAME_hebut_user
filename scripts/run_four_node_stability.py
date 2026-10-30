"""PyCharm: validate seeds 0,1,2 before the full four-task repeat grid."""
import sys
from pathlib import Path
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from came_repro.corrected_node_suite import main
if __name__ == '__main__':
    sys.argv.append('--stability-check')
    raise SystemExit(main())
