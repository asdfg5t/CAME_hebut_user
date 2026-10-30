"""Start the verified project runtime before importing any experiment dependencies."""
import os
from pathlib import Path
import subprocess
import sys


def ensure_project_environment(entry_file):
    entry = Path(entry_file).resolve()
    project = entry.parents[1]
    expected = project / '.venv-came39/Scripts/python.exe'
    if Path(sys.executable).resolve() == expected.resolve():
        return
    if not expected.is_file():
        raise RuntimeError('Verified project environment is missing: ' + str(expected)
                           + '. Restore it with scripts/bootstrap.ps1; no global install or CPU fallback was attempted.')
    print('[CAME] Launcher Python: ' + sys.executable, flush=True)
    print('[CAME] Starting verified experiment Python: ' + str(expected), flush=True)
    print('[CAME] Arguments preserved; model training still requires CUDA.', flush=True)
    command = [str(expected), '-u', str(entry), *sys.argv[1:]]
    process = subprocess.Popen(command, cwd=os.getcwd(),
                               env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'})
    try:
        code = process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise SystemExit(130)
    raise SystemExit(code)
