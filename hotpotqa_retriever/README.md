# Retrieval Server

## Setup

### 1. Create Environment
```bash
conda create -n retrieval_env python=3.10 -y
conda activate retrieval_env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build HotpotQA Corpus (AutoCoA-compatible)
This corpus is **not included** in the repo because it is large (hundreds of MB).
We build it from the original HotpotQA train + dev JSONL files by aggregating the
provided context (supporting-facts context with distractors) into document-level entries.

```bash
python build_hotpotqa_corpus.py \
  --train /path/to/hotpotqa_train.jsonl \
  --dev /path/to/hotpotqa_dev.jsonl \
  --output hotpotqa_corpus.json
```

### 4. Create FAISS Index
```bash
python create_faiss_index.py
```
This expects `hotpotqa_corpus.json` in the same folder (JSON list of `{id, contents}` objects).

### 5. Start Server
```bash
python run_server.py
```

## Usage

### Single Query
```bash
curl -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "tok_k": 3, "return_score": true}'
```

### Batch Query
```bash
curl -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": ["What is the capital of France?", "Who is the president?"], "tok_k": 2, "return_score": true}'
```

## Files

- `fast_retrieval_server.py` - Main server code
- `run_server.py` - Server launcher
- `build_hotpotqa_corpus.py` - Build supporting-facts corpus from HotpotQA train+dev
- `create_faiss_index.py` - Index creation script
- `requirements.txt` - Dependencies
- `hotpotqa_corpus.json` - Document corpus (507K documents)
- `hotpotqa_index.faiss` - FAISS index (created by step 4)
