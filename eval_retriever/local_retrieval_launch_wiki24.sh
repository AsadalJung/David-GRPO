#!/bin/bash

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$ROOT_DIR/.." && pwd)

echo "retriever use python:"
which python

DEFAULT_INDEX="$REPO_DIR/data/wiki-24/index/e5_Flat.index"
DEFAULT_CORPUS="$REPO_DIR/data/wiki-24/wiki-24.jsonl"

# Fallback to sibling Tree-GRPO data if local data is missing
if [[ ! -f "$DEFAULT_INDEX" && -f "$REPO_DIR/../Tree-GRPO/data/wiki-24/index/e5_Flat.index" ]]; then
  DEFAULT_INDEX="$REPO_DIR/../Tree-GRPO/data/wiki-24/index/e5_Flat.index"
fi
if [[ ! -f "$DEFAULT_CORPUS" && -f "$REPO_DIR/../Tree-GRPO/data/wiki-24/wiki-24.jsonl" ]]; then
  DEFAULT_CORPUS="$REPO_DIR/../Tree-GRPO/data/wiki-24/wiki-24.jsonl"
fi

INDEX_PATH=${INDEX_PATH:-$DEFAULT_INDEX}
CORPUS_PATH=${CORPUS_PATH:-$DEFAULT_CORPUS}
RETRIEVER_NAME=${RETRIEVER_NAME:-e5}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-intfloat/e5-base-v2}
TOPK=${TOPK:-3}
RETRIEVER_PORT=${RETRIEVER_PORT:-8004}
FAISS_GPU_FLAG=${FAISS_GPU_FLAG:-}

if [[ ! -f "$INDEX_PATH" ]]; then
  echo "ERROR: index file not found at $INDEX_PATH"
  echo "Set INDEX_PATH env or place index under $REPO_DIR/data/wiki-24/index/e5_Flat.index"
  exit 1
fi

if [[ ! -f "$CORPUS_PATH" ]]; then
  echo "ERROR: corpus file not found at $CORPUS_PATH"
  echo "Set CORPUS_PATH env or place corpus under $REPO_DIR/data/wiki-24/wiki-24.jsonl"
  exit 1
fi

python "$ROOT_DIR/search_r1/search/retrieval_server.py" --index_path "$INDEX_PATH" \
                                                      --corpus_path "$CORPUS_PATH" \
                                                      --topk "$TOPK" \
                                                      --retriever_name "$RETRIEVER_NAME" \
                                                      --retriever_model "$RETRIEVER_MODEL" \
                                                      --port "$RETRIEVER_PORT" \
                                                      $FAISS_GPU_FLAG
