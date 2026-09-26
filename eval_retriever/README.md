# Eval Retriever (wiki-18 / wiki-24)

This folder contains the local retrieval server used **for evaluation**.
- **wiki-18** (default port **8003**) for standard eval
- **wiki-24** (default port **8004**) for AntiLeakBench eval

> Note: Training uses the HotpotQA retriever (port 8001) under `hotpotqa_retriever/`.

## Contents
- `local_retrieval_launch.sh` — launch wiki-18 server (port 8003)
- `local_retrieval_launch_wiki24.sh` — launch wiki-24 server (port 8004)
- `search_r1/` — retrieval server implementation
- `requirements.txt` — deps (aligned to Tree-GRPO retriever env)
- `scripts/prepare_wiki18.sh` — download & prepare wiki-18 index/corpus
- `scripts/download_wiki18.py` — wiki-18 download helper (from Tree-GRPO)
- `scripts/build_faiss_index.py` — build FAISS index from a JSONL corpus (for wiki-24)

## Data layout (expected)
```
David-GRPO/
  data/
    wiki/
      e5_Flat.index
      wiki-18.jsonl
    wiki-24/
      index/e5_Flat.index
      wiki-24.jsonl
```

Set your repo root once:
```bash
export REPO_ROOT=/path/to/David-GRPO
```

You can also override paths at runtime:
```bash
INDEX_PATH=/path/to/e5_Flat.index \
CORPUS_PATH=/path/to/wiki-18.jsonl \
${REPO_ROOT}/eval_retriever/local_retrieval_launch.sh
```

## 1) Install deps
```bash
python -m pip install -r ${REPO_ROOT}/eval_retriever/requirements.txt
```
If you **don’t** have GPU FAISS, replace `faiss-gpu` with `faiss-cpu` in `requirements.txt`.

## 2) Prepare corpora (step-by-step)
### A) wiki-18 (public, prebuilt index)
```bash
# downloads index shards + corpus, merges index, unzips corpus
${REPO_ROOT}/eval_retriever/scripts/prepare_wiki18.sh
```
By default this writes to:
```
${REPO_ROOT}/data/wiki/
```
You can override with:
```bash
SAVE_PATH=/path/to/save/wiki \
${REPO_ROOT}/eval_retriever/scripts/prepare_wiki18.sh
```

### B) wiki-24 (your own dump)
1) Put your `wiki-24.jsonl` here (or keep elsewhere and use env vars):
```
${REPO_ROOT}/data/wiki-24/wiki-24.jsonl
```
2) Build the FAISS index:
```bash
python ${REPO_ROOT}/eval_retriever/scripts/build_faiss_index.py \
  --corpus ${REPO_ROOT}/data/wiki-24/wiki-24.jsonl \
  --output ${REPO_ROOT}/data/wiki-24/index/e5_Flat.index \
  --model intfloat/e5-base-v2 \
  --batch-size 256
```

## 3) Launch retriever
```bash
# wiki-18 (standard eval)
${REPO_ROOT}/eval_retriever/local_retrieval_launch.sh

# wiki-24 (AntiLeakBench)
${REPO_ROOT}/eval_retriever/local_retrieval_launch_wiki24.sh
```

## Environment variables
- `INDEX_PATH`, `CORPUS_PATH` — override data paths
- `RETRIEVER_PORT` — change port (default 8003/8004)
- `TOPK` — number of documents returned (default 3)
- `RETRIEVER_MODEL` — encoder name (default `intfloat/e5-base-v2`)
- `FAISS_GPU_FLAG` — optional FAISS GPU args
