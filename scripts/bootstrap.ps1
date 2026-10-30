$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = 'C:\Users\DELL\AppData\Local\Programs\Python\Python39\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'User Python 3.9.13 was not found.' }
if (-not (Test-Path '.venv-came39/Scripts/python.exe')) {
    & $taskPython -m venv .venv-came39
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& ./.venv-came39/Scripts/python.exe -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed' }
& ./.venv-came39/Scripts/python.exe -m pip install 'torch==2.1.2+cu118' --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
$taskRequirements = if (Test-Path 'requirements/phase_02_windows-cu118.lock.txt') { 'requirements/phase_02_windows-cu118.lock.txt' } elseif (Test-Path 'requirements/windows-cu118.lock.txt') { 'requirements/windows-cu118.lock.txt' } else { 'requirements/windows-cu118.txt' }
& ./.venv-came39/Scripts/python.exe -m pip install -r $taskRequirements --extra-index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw 'Dependencies installation failed' }
& ./.venv-came39/Scripts/python.exe -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
