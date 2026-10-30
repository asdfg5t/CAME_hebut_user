"""Historical name; redirects to the genuine official-data workflow."""
from pathlib import Path
import sys
if __name__ == '__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
    from run_21_goodreads_official import main
    main()
