param(
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('all','calibrate','tune','main','audit')][string]$Phase = 'all',
    [string]$Python = 'E:\envs\lorara_realmodel\Scripts\python.exe',
    [switch]$Smoke
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment not found: $Python" }
$oldMkl = $env:MKL_NUM_THREADS
$oldOpenblas = $env:OPENBLAS_NUM_THREADS
Push-Location -LiteralPath $projectPath
try {
    $env:MKL_NUM_THREADS = '4'
    $env:OPENBLAS_NUM_THREADS = '4'
    $arguments = @('-u', 'scripts/realmodel_experiment.py', '--run', $Output, '--phase', $Phase)
    if ($Smoke) { $arguments += '--smoke' }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Real-model experiment failed with exit code $LASTEXITCODE" }
} finally {
    $env:MKL_NUM_THREADS = $oldMkl
    $env:OPENBLAS_NUM_THREADS = $oldOpenblas
    Pop-Location
}
