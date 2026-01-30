#!/usr/bin/env python3
"""
Build HotpotQA retriever corpus by aggregating HotpotQA train+dev contexts
into document-level entries (AutoCoA-compatible format).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Iterator
from itertools import chain


def iter_hotpot_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_context(record: dict) -> List[Tuple[str, List[str]]]:
    """Return list of (title, sentences) pairs from record.context."""
    ctx = record.get("context")
    if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        return list(zip(titles, sentences))
    if isinstance(ctx, list):
        # Original HotpotQA format: list of [title, sentences]
        pairs = []
        for item in ctx:
            if not item or len(item) != 2:
                continue
            title, sents = item
            pairs.append((title, sents))
        return pairs
    return []


def normalize_sentences(sentences: List[str]) -> str:
    """Normalize sentence spacing to match AutoCoA corpus formatting."""
    if not sentences:
        return ""
    parts: List[str] = [sentences[0].lstrip()]
    parts.extend(sentences[1:])
    return "".join(parts)


def iter_entries(records: Iterable[dict]) -> Iterator[dict]:
    """
    AutoCoA-compatible corpus generator:
    - iterate train+dev in order
    - for each context doc, if title not seen, yield entry
    - contents = \"<title>\" + '\\n' + ''.join(sentences)
    """
    seen = set()
    for record in records:
        for title, sents in extract_context(record):
            if title in seen:
                continue
            seen.add(title)
            contents = f"\"{title}\"\n" + normalize_sentences(sents)
            contents = contents.rstrip()
            if not contents:
                continue
            yield {"id": title, "contents": contents}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hotpotqa_corpus.json from HotpotQA train+dev supporting facts."
    )
    parser.add_argument("--train", required=True, help="Path to HotpotQA train JSONL")
    parser.add_argument("--dev", required=True, help="Path to HotpotQA dev JSONL")
    parser.add_argument(
        "--output",
        default="hotpotqa_corpus.json",
        help="Output JSON file (list of {id, contents})",
    )

    args = parser.parse_args()

    train_path = Path(args.train)
    dev_path = Path(args.dev)
    out_path = Path(args.output)

    train_iter = iter_hotpot_jsonl(train_path)
    dev_iter = iter_hotpot_jsonl(dev_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        count = 0
        for entry in iter_entries(chain(train_iter, dev_iter)):
            if not first:
                f.write(",\n")
            entry_json = json.dumps(entry, ensure_ascii=False, indent=2)
            entry_json = "\n".join("  " + line for line in entry_json.split("\n"))
            f.write(entry_json)
            first = False
            count += 1
        f.write("\n]")

    print(f"Wrote {count} docs to {out_path}")


if __name__ == "__main__":
    main()
