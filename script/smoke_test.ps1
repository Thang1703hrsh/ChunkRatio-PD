param([string]$Output = 'output/smoke')
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    python -m chunkratio smoke --output $Output
    if ($LASTEXITCODE -ne 0) { throw "CRPD smoke failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
