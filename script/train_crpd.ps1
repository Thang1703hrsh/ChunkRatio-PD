param([Parameter(Mandatory=$true)][string]$Config)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    python -m chunkratio validate-config --config $Config
    if ($LASTEXITCODE -ne 0) { throw 'Invalid CRPD config' }
    python train.py --crpd evaluate --config $Config
    if ($LASTEXITCODE -ne 0) { throw 'CRPD pipeline failed' }
} finally { Pop-Location }
