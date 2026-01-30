import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Download wiki-18 index shards and corpus from Hugging Face.")
parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")

args = parser.parse_args()

# Same repos as Tree-GRPO/scripts/download.py
repo_id = "PeterJinGo/wiki-18-e5-index"
for file in ["part_aa", "part_ab"]:
    hf_hub_download(
        repo_id=repo_id,
        filename=file,
        repo_type="dataset",
        local_dir=args.save_path,
    )

repo_id = "PeterJinGo/wiki-18-corpus"
hf_hub_download(
    repo_id=repo_id,
    filename="wiki-18.jsonl.gz",
    repo_type="dataset",
    local_dir=args.save_path,
)
