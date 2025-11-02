#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="dist"
if [ $# -gt 0 ]; then
  OUTPUT_DIR=$1
  shift
fi

mkdir -p "$OUTPUT_DIR"

python -m nuitka \
  --module api.py \
  --include-module=train \
  --include-module=deploy \
  --include-module=encrypt \
  --include-package=models \
  --include-package=seal \
  --output-dir="$OUTPUT_DIR" \
  "$@"

echo "Nuitka build finished. Compiled module located in $OUTPUT_DIR."
