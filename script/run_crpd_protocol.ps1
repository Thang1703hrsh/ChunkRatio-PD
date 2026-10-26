param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    python script/run_crpd_protocol.py @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'CRPD protocol command failed' }
} finally { Pop-Location }
