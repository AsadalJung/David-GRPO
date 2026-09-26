"""Read-only duplicate audit of a raw four-field synthetic QA JSONL."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata


def normalized(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", html.unescape(str(value))).casefold()).strip()


def compact(value):
    return re.sub(r"[\W_]+", "", normalized(value))


def audit(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    keys = {"id": [], "normalized_question": [], "compact_question": [],
            "normalized_qa": [], "evidence_bundle": []}
    doc_counts = Counter()
    within_row_duplicates = 0
    for row in rows:
        assert set(row) == {"id", "question", "answer", "evidence_documents"}
        assert all(set(doc) == {"id", "title", "content"} for doc in row["evidence_documents"])
        keys["id"].append(str(row["id"]))
        keys["normalized_question"].append(normalized(row["question"]))
        keys["compact_question"].append(compact(row["question"]))
        keys["normalized_qa"].append((normalized(row["question"]), normalized(row["answer"])))
        docs = [(normalized(doc["title"]), normalized(doc["content"])) for doc in row["evidence_documents"]]
        within_row_duplicates += len(docs) - len(set(docs))
        keys["evidence_bundle"].append(tuple(sorted(docs)))
        doc_counts[len(docs)] += 1
    duplicates = {key: len(values) - len(set(values)) for key, values in keys.items()}
    return {"file": path.name, "rows": len(rows), "sha256": hashlib.sha256(raw).hexdigest(),
            "duplicate_extra_rows": duplicates, "within_row_duplicate_documents": within_row_duplicates,
            "document_count_distribution": dict(sorted(doc_counts.items())),
            "passed": not any(duplicates.values()) and not within_row_duplicates}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    results = [audit(path) for path in args.inputs]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(item["passed"] for item in results) else 1)
