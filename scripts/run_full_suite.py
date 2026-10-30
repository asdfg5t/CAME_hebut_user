"""Default: corrected four NC tasks; legacy grid requires --legacy-full-suite."""
import sys
from pathlib import Path
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
def main():
    if '--legacy-full-suite' in sys.argv:
        sys.argv.remove('--legacy-full-suite')
        from came_repro.full_suite import main as run
    else:
        if len(sys.argv) == 1:
            from came_repro.completed_four_results import show_if_completed
            completed = show_if_completed()
            if completed is not None:
                return completed
        from came_repro.corrected_node_suite import main as run
    return run()
if __name__ == '__main__':
    raise SystemExit(main())
