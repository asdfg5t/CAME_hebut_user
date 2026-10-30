param(
    [ValidateSet('representation','few_shot')][string]$Protocol = 'representation',
    [int]$Seed = 42,
    [string]$ResumeCheckpoint = '',
    [int]$SmokeSteps = 0,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ProjectPython = Join-Path $ProjectRoot '.venv-came39/Scripts/python.exe'
$Profile = Join-Path $ProjectRoot "configs/joint_${Protocol}_reconstruction.json"
Push-Location -LiteralPath $ProjectRoot
try {
    & $ProjectPython scripts/check_storage_budget.py --additional-cache-gib 2
    if ($LASTEXITCODE -ne 0) { throw 'Storage budget check failed' }
    $TrainArguments = @('-u','-m','came_repro.joint_train','--config',$Profile,'--seed',"$Seed")
    if ($ResumeCheckpoint) { $TrainArguments += @('--resume',$ResumeCheckpoint) }
    if ($SmokeSteps -gt 0) { $TrainArguments += @('--max-steps',"$SmokeSteps") }
    if ($PlanOnly) {
        Write-Output "Interpreter: $ProjectPython"
        Write-Output "Arguments: $($TrainArguments -join ' ')"
        return
    }
    & $ProjectPython @TrainArguments
    if ($LASTEXITCODE -ne 0) { throw 'Training stopped; inspect the new run logs/status and latest consistent checkpoint' }
} finally {
    Pop-Location
}
