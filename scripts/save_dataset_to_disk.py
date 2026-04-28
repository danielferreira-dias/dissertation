"""Build a self-contained HF datasets folder on local disk for upload to Unsloth Studio.

Usage:
    python scripts/save_dataset_to_disk.py --format full_reasoning
    python scripts/save_dataset_to_disk.py --format label_only

Output goes to:
    ./dataset_export/derm-<format>-with-images/
        ├── dataset_info.json
        ├── train/<parquet shards>
        └── val/<parquet shards>

This folder can be:
  - Loaded with `datasets.load_from_disk("./dataset_export/derm-…/")`
  - Uploaded to Unsloth Studio's "Upload" button (zip first if Studio expects an archive)
  - rsync'd to a pod and loaded the same way
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from upload_dataset_to_hub import (  # type: ignore
    DATA_FT,
    build_source_map,
    load_and_augment,
)

OUT_ROOT = REPO_ROOT / "dataset_export"


def build_and_save(fmt: str) -> Path:
    from datasets import Dataset, DatasetDict, Features, Image, Value

    out_dir = OUT_ROOT / f"derm-{fmt}-with-images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building source-attribution map ...")
    source_map = build_source_map()
    print(f"Indexed {len(source_map)} basenames.\n")

    splits = {}
    for split in ("train", "val"):
        jp = DATA_FT / fmt / f"{split}.jsonl"
        rows = load_and_augment(jp, source_map)

        # Replace relative paths with absolute (so Image() can read them on save).
        kept = []
        n_skipped = 0
        for r in rows:
            abs_path = REPO_ROOT / r["image"]
            if not abs_path.exists() or abs_path.stat().st_size == 0:
                n_skipped += 1
                continue
            r = dict(r)
            r["image"] = str(abs_path)
            kept.append(r)
        print(f"  {split}: {len(kept)} rows ({n_skipped} skipped)")

        # Schema: top-level Image() embeds bytes; messages keep path strings for traceability.
        features = Features({
            "messages": Dataset.from_list(kept[:1]).features["messages"],
            "image": Image(),
            "image_id": Value("string"),
            "class": Value("string"),
            "source": Value("string"),
        })
        splits[split] = Dataset.from_list(kept, features=features)

    dd = DatasetDict(splits)
    print(f"\nSaving to {out_dir} (this materialises image bytes — ~12 GB on disk)...")
    dd.save_to_disk(str(out_dir))
    print(f"Done.")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=["full_reasoning", "label_only"], required=True)
    args = p.parse_args()
    out = build_and_save(args.format)
    print(f"\nUpload this folder to Unsloth Studio:")
    print(f"  {out}")
    print(f"\nOr zip first:")
    print(f"  cd {out.parent} && zip -r derm-{args.format}-with-images.zip derm-{args.format}-with-images/")


if __name__ == "__main__":
    main()
