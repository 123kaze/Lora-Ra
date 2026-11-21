param([Parameter(Mandatory=$true)][string]$Output,[switch]$Resume,[switch]$Smoke)
$ErrorActionPreference='Stop'
$previousPythonPath=$env:PYTHONPATH
$previousWorkspace=$env:CUBLAS_WORKSPACE_CONFIG
try {
    $env:PYTHONPATH=Join-Path (Split-Path -Parent $PSScriptRoot) 'src'
    $env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
    $arguments=@('-m','carryrank.diagnostic','--output',$Output)
    if ($Resume) { $arguments+='--resume' }
    if ($Smoke) { $arguments+='--smoke' }
    & 'E:\envs\LangChain_agent\python.exe' @arguments
    if ($LASTEXITCODE -ne 0) { throw "Diagnostic failed: $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH=$previousPythonPath
    $env:CUBLAS_WORKSPACE_CONFIG=$previousWorkspace
}
