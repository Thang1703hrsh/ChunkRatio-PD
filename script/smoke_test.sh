#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUTPUT=output/smoke
if [ "${1:-}" = "--output" ]; then OUTPUT="$2"; fi
python -m chunkratio smoke --output "$OUTPUT"
python -m chunkratio smoke --protocol narval-tiny --output "$OUTPUT/narval-tiny"
