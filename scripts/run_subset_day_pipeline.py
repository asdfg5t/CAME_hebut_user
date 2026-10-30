"""PyCharm one-click entry: offline PPR, fixed subset rounds, quality-first 24h plan."""
import sys
from pathlib import Path
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from came_repro.subset_day_pipeline import main
if __name__ == '__main__':
    main()
