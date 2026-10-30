"""Serial one-seed Sports/Cloth/Goodreads-LP diagnostics, not ten-seed repeats."""
import sys
from pathlib import Path
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from came_repro.remaining_lp_validation import main
if __name__ == '__main__':
    main()
