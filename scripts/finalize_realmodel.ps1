param(
    [Parameter(Mandatory=$true)][string]$Run,
    [string]$CpuPython = 'E:\Conda\python.exe',
    [string]$GpuPython = 'E:\envs\lorara_realmodel\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectPath
$oldMkl = $env:MKL_NUM_THREADS
$oldOpenblas = $env:OPENBLAS_NUM_THREADS
try {
    $runPath = (Resolve-Path -LiteralPath $Run).Path
    $completion = Get-Content -LiteralPath (Join-Path $runPath 'completion.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $completion.complete -or $completion.smoke) { throw 'A completed nonsmoke run is required' }
    $runName = Split-Path -Leaf $runPath
    $bundle = Join-Path 'results' $runName
    $env:MKL_NUM_THREADS = '4'
    $env:OPENBLAS_NUM_THREADS = '4'
    if (-not (Test-Path -LiteralPath (Join-Path $runPath 'calibration_cost_profile.json'))) {
        & $GpuPython -u scripts/profile_realmodel_calibration.py $runPath
        if ($LASTEXITCODE -ne 0) { throw 'Calibration resource/reload probe failed' }
    }
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    & $CpuPython scripts/analyze_realmodel.py $runPath
    if ($LASTEXITCODE -ne 0) { throw 'Independent metric analysis failed' }
    if (-not (Test-Path -LiteralPath $bundle)) {
        & $CpuPython scripts/export_realmodel.py $runPath
        if ($LASTEXITCODE -ne 0) { throw 'Evidence export failed' }
    }
    & $CpuPython figures/realmodel/plot_results.py $bundle
    if ($LASTEXITCODE -ne 0) { throw 'Real-data figure generation failed' }
    & $CpuPython scripts/report_realmodel.py $bundle
    if ($LASTEXITCODE -ne 0) { throw 'Measured report generation failed' }
    & $CpuPython scripts/export_realmodel.py $bundle --manifest-only
    if ($LASTEXITCODE -ne 0) { throw 'Evidence sealing failed' }
    & $CpuPython scripts/verify_realmodel_evidence.py $bundle
    if ($LASTEXITCODE -ne 0) { throw 'Independent NumPy evidence verification failed' }
} finally {
    $env:MKL_NUM_THREADS = $oldMkl
    $env:OPENBLAS_NUM_THREADS = $oldOpenblas
    Pop-Location
}
