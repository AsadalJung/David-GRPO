#!/usr/bin/env python3
import argparse
import json
import os
from typing import List

import numpy as np
import faiss
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm


def load_model(model_path: str):
    AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask, method="mean"):
    if method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if method == "cls":
        return last_hidden_state[:, 0]
    if method == "pooler":
        return pooler_output
    raise NotImplementedError("Pooling method not implemented")


def encode_batch(tokenizer, model, texts: List[str], max_length: int, pooling_method: str, model_name: str):
    if "e5" in model_name.lower():
        texts = [f"passage: {t}" for t in texts]

    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs, return_dict=True)
        emb = pooling(output.pooler_output, output.last_hidden_state, inputs["attention_mask"], pooling_method)
        emb = torch.nn.functional.normalize(emb, dim=-1)

    emb = emb.detach().cpu().numpy().astype(np.float32, order="C")
    return emb


def iter_jsonl(corpus_path: str, text_field: str):
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if text_field:
                text = obj.get(text_field, "")
            else:
                text = obj.get("contents") or obj.get("text") or obj.get("passage") or ""
            yield text


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from a JSONL corpus (E5-style embeddings)")
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus file")
    parser.add_argument("--output", required=True, help="Output FAISS index path")
    parser.add_argument("--model", default="intfloat/e5-base-v2", help="HF model name or path")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length")
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls", "pooler"], help="Pooling method")
    parser.add_argument("--text-field", default="", help="Explicit text field in JSONL (optional)")

    args = parser.parse_args()

    if not os.path.isfile(args.corpus):
        raise FileNotFoundError(f"Corpus not found: {args.corpus}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    model, tokenizer = load_model(args.model)

    index = None
    buffer = []
    total = 0

    for text in tqdm(iter_jsonl(args.corpus, args.text_field), desc="Reading corpus"):
        buffer.append(text)
        if len(buffer) >= args.batch_size:
            emb = encode_batch(tokenizer, model, buffer, args.max_length, args.pooling, args.model)
            if index is None:
                index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            total += emb.shape[0]
            buffer = []

    if buffer:
        emb = encode_batch(tokenizer, model, buffer, args.max_length, args.pooling, args.model)
        if index is None:
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        total += emb.shape[0]

    if index is None:
        raise RuntimeError("No embeddings were generated. Check the corpus format.")

    faiss.write_index(index, args.output)
    print(f"Saved FAISS index to {args.output} (vectors: {total})")


if __name__ == "__main__":
    main()
