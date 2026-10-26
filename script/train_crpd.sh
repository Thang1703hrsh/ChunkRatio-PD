#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:?Provide a resolved config/crpd protocol JSON}"
python -m chunkratio validate-config --config "$CONFIG"
python train.py --crpd evaluate --config "$CONFIG"
