#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_DIR=$(cd "$ROOT_DIR/.." && pwd)

SAVE_PATH=${SAVE_PATH:-"$REPO_DIR/data/wiki"}

mkdir -p "$SAVE_PATH"

python "$ROOT_DIR/scripts/download_wiki18.py" --save_path "$SAVE_PATH"

# Merge FAISS shards
cat "$SAVE_PATH"/part_* > "$SAVE_PATH"/e5_Flat.index

# Unzip corpus if needed
if [[ -f "$SAVE_PATH/wiki-18.jsonl.gz" ]]; then
  gzip -d -f "$SAVE_PATH/wiki-18.jsonl.gz"
fi

echo "wiki-18 prepared under $SAVE_PATH"
