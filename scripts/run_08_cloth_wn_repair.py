"""PyCharm直接运行：从完整断点继续Cloth / WN有限修复验证，不重训已完成六项。"""
import sys
from pathlib import Path

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_environment import ensure_project_environment
    ensure_project_environment(__file__)
    from cloth_wn_repair_entry import main
    raise SystemExit(main())
