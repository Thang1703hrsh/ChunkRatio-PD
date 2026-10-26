#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python script/run_crpd_protocol.py "$@"
