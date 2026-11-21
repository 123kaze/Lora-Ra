param(
    [string]$Config = 'configs/pilot.json',
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('all','pretrain','main','sweep','report')][string]$Stage = 'all',
    [switch]$Resume,
    [string]$Python = 'E:\envs\LangChain_agent\python.exe'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$previousPythonPath = $env:PYTHONPATH
$previousWorkspaceConfig = $env:CUBLAS_WORKSPACE_CONFIG
$previousUnbuffered = $env:PYTHONUNBUFFERED
try {
    $env:PYTHONPATH = Join-Path $projectRoot 'src'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $env:PYTHONUNBUFFERED = '1'
    Push-Location $projectRoot
    $experimentArguments = @('-m','carryrank.experiment','--config',$Config,'--output',$Output,'--stage',$Stage)
    if ($Resume) { $experimentArguments += '--resume' }
    & $Python @experimentArguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
    $env:CUBLAS_WORKSPACE_CONFIG = $previousWorkspaceConfig
    $env:PYTHONUNBUFFERED = $previousUnbuffered
}
