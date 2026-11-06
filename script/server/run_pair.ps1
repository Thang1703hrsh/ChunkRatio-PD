# Windows launcher for script/server/run_pair.py (all arguments pass through). Run from anywhere.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..\..')
$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
& $python script/server/run_pair.py @args
exit $LASTEXITCODE
