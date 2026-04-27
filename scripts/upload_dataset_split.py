"""Re-publish the two `derm-reasoning` configs as standalone Hub repos.

The combined repo (danielfdias98/derm-reasoning) has both `full_reasoning` and
`label_only` configs under one dataset_id, which means downstream users have to
pass `config_name=` to load_dataset. For the ablation campaign it's nicer to
have two cleanly-separated repos that can be plugged into the trainer with no
config selection:

    danielfdias98/derm-reasoning-label-only      (just diagnosis + category)
    danielfdias98/derm-reasoning-full-reasoning  (diagnosis + structured reasoning)

Usage:
    HF_TOKEN=hf_… python scripts/upload_dataset_split.py
    # or one at a time:
    HF_TOKEN=hf_… python scripts/upload_dataset_split.py --only label_only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from upload_dataset_to_hub import (  # type: ignore
    DATA_FT,
    build_source_map,
    load_and_augment,
)

NEW_REPOS = {
    "label_only": "danielfdias98/derm-reasoning-label-only",
    "full_reasoning": "danielfdias98/derm-reasoning-full-reasoning",
}


def push_one(fmt: str, repo_id: str, source_map: dict[str, str]) -> None:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    print(f"\n=== {fmt} → {repo_id} ===")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    splits = {}
    for split in ("train", "val"):
        jp = DATA_FT / fmt / f"{split}.jsonl"
        rows = load_and_augment(jp, source_map)
        print(f"  {split}: {len(rows)} rows")
        splits[split] = Dataset.from_list(rows)
    DatasetDict(splits).push_to_hub(repo_id, token=os.environ.get("HF_TOKEN"))
    print(f"  pushed → https://huggingface.co/datasets/{repo_id}")


def upload_card(repo_id: str, fmt: str) -> None:
    """Upload a per-repo README.md with format-specific framing."""
    from huggingface_hub import HfApi

    is_full = fmt == "full_reasoning"
    title_fmt = "Full Reasoning" if is_full else "Label Only"
    desc_fmt = (
        "structured chain-of-thought clinical reasoning (diagnosis + observation + "
        "morphology + color + texture + border + location + size + reasoning)"
        if is_full
        else "diagnosis + clinical category labels only — the ablation baseline that "
        "tests whether the reasoning supervision in the full-reasoning variant actually helps"
    )
    sister_repo = NEW_REPOS["label_only"] if is_full else NEW_REPOS["full_reasoning"]

    body = f"""---
language: [en]
license: cc-by-nc-sa-4.0
task_categories: [image-text-to-text, visual-question-answering]
size_categories: [10K<n<100K]
tags: [medical, dermatology, vision-language, vlm-finetune, fairness]
pretty_name: Dermatology Reasoning Dataset — {title_fmt}
---

# Dermatology Reasoning Dataset — {title_fmt}

This is the **{fmt}** ablation variant of [danielfdias98/derm-reasoning](https://huggingface.co/datasets/danielfdias98/derm-reasoning), republished as a standalone repo so it can be plugged directly into a trainer without picking a config.

The assistant turn for every row contains: **{desc_fmt}**.

## Quick load

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}")
print(ds)  # DatasetDict {{train: 25_637, val: 2_849}}
```

## Sister repo

The complementary ablation variant is at [{sister_repo}](https://huggingface.co/datasets/{sister_repo}). Same train/val split (95/5, seed=42), same images, just different assistant-turn density.

## Image sources

The combined parent repo has the full per-source attribution table and image-redistribution split (public-redistributable + private-full image sets). See [danielfdias98/derm-reasoning](https://huggingface.co/datasets/danielfdias98/derm-reasoning) for the canonical README, citations, and `download_images.py` script.

## License

CC-BY-NC-SA 4.0 (matches the most restrictive component of the source datasets, SkinCAP). Non-commercial research use only.

## Citation

```bibtex
@misc{{dias2026derm-reasoning,
  author = {{Ferreira Dias, Daniel}},
  title  = {{Dermatology Reasoning Dataset: Structured chain-of-thought annotations across five public sources}},
  year   = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/danielfdias98/derm-reasoning}}}},
}}
```
"""

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=body.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"README for {title_fmt} variant",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=list(NEW_REPOS.keys()),
                   help="Push just one of the two splits (default: both)")
    args = p.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("ERROR: HF_TOKEN env var not set. export a write-scope token and re-run.")

    print("Building source-attribution map ...")
    source_map = build_source_map()
    print(f"Indexed {len(source_map)} basenames.\n")

    for fmt, repo in NEW_REPOS.items():
        if args.only and args.only != fmt:
            continue
        push_one(fmt, repo, source_map)
        upload_card(repo, fmt)


if __name__ == "__main__":
    main()
