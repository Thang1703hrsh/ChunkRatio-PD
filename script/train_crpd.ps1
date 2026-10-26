param([Parameter(Mandatory=$true)][string]$Config)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    # Native stderr warnings must not abort when callers redirect them; exit codes decide.
    $ErrorActionPreference = 'Continue'
    python -m chunkratio validate-config --config $Config
    if ($LASTEXITCODE -ne 0) { throw 'Invalid CRPD config' }
    python train.py --crpd evaluate --config $Config
    if ($LASTEXITCODE -ne 0) { throw 'CRPD pipeline failed' }
} finally { Pop-Location }
