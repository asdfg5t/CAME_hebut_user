"""Validation-only KG graph audit; complete test requires a frozen selection lock."""
import sys
from pathlib import Path
if __name__ == '__main__':
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from came_repro.kg_graph_probe import main
if __name__ == '__main__':
    main()
