#!/usr/bin/env bash
# Linux launcher for script/server/run_pair.py (all arguments pass through). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PYTHON:-python}" script/server/run_pair.py "$@"
