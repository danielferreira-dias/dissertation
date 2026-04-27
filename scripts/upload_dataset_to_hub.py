"""Upload the dermatology reasoning dataset to HuggingFace Hub.

Three artifacts:
  --target reasoning      → public  danielfdias98/derm-reasoning
                             (JSONLs only, with source attribution per row)
  --target redistributable → public  danielfdias98/derm-reasoning-redistributable
                             (SCIN + PAD-UFES-20 images only, both CC-compatible)
  --target full           → private danielfdias98/derm-reasoning-full
                             (all images + JSONLs, examiner access only)

Source attribution is computed by walking data/dataset/<source>/ folders and
indexing image basenames; the per-source folders are the source-of-truth from
Phase 1 of the dissertation. Train images that don't appear in any per-source
folder fall back to filename heuristics (PAT_* → pad_ufes, descriptive → kaggle_dermnet).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FT = REPO_ROOT / "data" / "fine_tune"
DATA_DATASET = REPO_ROOT / "data" / "dataset"
FINAL_TRAIN = REPO_ROOT / "final" / "train"

# Per-source folders under data/dataset/, in priority order for ambiguous cases.
SOURCES = ["scin", "pad_ufes", "skincap", "kaggle_dermnet", "dermnet_nz"]

# Sources whose license clearly permits public Hub redistribution.
REDISTRIBUTABLE = {"scin", "pad_ufes"}


def build_source_map() -> dict[str, str]:
    """Return {basename_without_ext: source} by walking data/dataset/<source>/."""
    m: dict[str, str] = {}
    for src in SOURCES:
        d = DATA_DATASET / src
        if not d.is_dir():
            continue
        n = 0
        for cls_dir in d.iterdir():
            if not cls_dir.is_dir():
                continue
            for img in cls_dir.iterdir():
                if img.is_file() and img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    key = img.stem
                    # First source seen wins; SOURCES is in priority order.
                    if key not in m:
                        m[key] = src
                        n += 1
        print(f"  indexed {n:>6} files from {src}")
    return m


def fallback_source(filename: str) -> str:
    base = Path(filename).stem
    if base.startswith("PAT_"):
        return "pad_ufes"
    # numeric-only and hash-style filenames are usually scraped (DermNet variants)
    if base.lstrip("-").isdigit():
        return "dermnet_nz"
    # Descriptive names (e.g. "basal-cell-carcinoma-face-1") are Kaggle DermNet.
    return "kaggle_dermnet"


def attribute(image_path: str, source_map: dict[str, str]) -> str:
    base = Path(image_path).stem
    return source_map.get(base) or fallback_source(image_path)


def _normalize_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    """Make every message's `content` a list of text/image parts (PyArrow requires
    a homogeneous schema). Returns the normalized messages plus the first image path.
    """
    out: list[dict[str, Any]] = []
    image_path: str | None = None
    for m in messages:
        content = m.get("content")
        parts: list[dict[str, Any]] = []
        if isinstance(content, list):
            for c in content:
                if c.get("type") == "image":
                    img = c.get("image")
                    if image_path is None:
                        image_path = img
                    parts.append({"type": "image", "text": "", "image": img or ""})
                else:
                    parts.append({"type": "text", "text": c.get("text", ""), "image": ""})
        else:
            # String content (assistant turn) → single text part
            parts.append({"type": "text", "text": str(content), "image": ""})
        out.append({"role": m["role"], "content": parts})
    return out, image_path


def load_and_augment(jsonl_path: Path, source_map: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in jsonl_path.open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        norm_msgs, image_path = _normalize_messages(d.get("messages", []))
        if not image_path:
            continue
        cls = Path(image_path).parent.name
        src = attribute(image_path, source_map)
        rows.append({
            "messages": norm_msgs,
            "image": image_path,
            "image_id": Path(image_path).stem,
            "class": cls,
            "source": src,
        })
    return rows


def _features():
    """Explicit schema so PyArrow doesn't have to infer mixed types."""
    from datasets import Features, Sequence, Value
    part = {"type": Value("string"), "text": Value("string"), "image": Value("string")}
    message = {"role": Value("string"), "content": Sequence(part)}
    return Features({
        "messages": Sequence(message),
        "image": Value("string"),
        "image_id": Value("string"),
        "class": Value("string"),
        "source": Value("string"),
    })


def push_reasoning(repo_id: str, source_map: dict[str, str]) -> None:
    """Build datasets.DatasetDict with two configs × two splits and push (public)."""
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    print(f"\n=== Pushing reasoning dataset → {repo_id} ===")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    for fmt in ("full_reasoning", "label_only"):
        print(f"\n[{fmt}]")
        splits = {}
        for split in ("train", "val"):
            jp = DATA_FT / fmt / f"{split}.jsonl"
            print(f"  loading {jp}")
            rows = load_and_augment(jp, source_map)
            print(f"  {len(rows)} rows")
            ds = Dataset.from_list(rows)
            splits[split] = ds
        dd = DatasetDict(splits)
        # Show source distribution
        srcs = collections.Counter(r["source"] for r in dd["train"])
        print(f"  source distribution (train): {dict(srcs)}")

        print(f"  pushing config={fmt}")
        dd.push_to_hub(repo_id, config_name=fmt)


def push_redistributable(repo_id: str, source_map: dict[str, str]) -> None:
    """Push only SCIN + PAD-UFES-20 images as a Hub dataset (public, CC-compatible)."""
    from datasets import Dataset, Features, Image, Value
    from huggingface_hub import HfApi

    print(f"\n=== Pushing redistributable images → {repo_id} ===")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for cls_dir in FINAL_TRAIN.iterdir():
        if not cls_dir.is_dir():
            continue
        for img in cls_dir.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            src = attribute(str(img.relative_to(REPO_ROOT)), source_map)
            if src not in REDISTRIBUTABLE:
                n_skipped += 1
                continue
            if img.stat().st_size == 0:
                continue
            rows.append({
                "image": str(img),  # datasets.Image() will load on push
                "image_id": img.stem,
                "class": cls_dir.name,
                "source": src,
            })
    print(f"  {len(rows)} redistributable images (skipped {n_skipped} non-CC)")

    features = Features({
        "image": Image(),
        "image_id": Value("string"),
        "class": Value("string"),
        "source": Value("string"),
    })
    ds = Dataset.from_list(rows, features=features)
    print(f"  pushing")
    ds.push_to_hub(repo_id)


def push_full_private(repo_id: str, source_map: dict[str, str]) -> None:
    """Upload the full image set as a Parquet/Image dataset + all four JSONLs as
    sibling reasoning configs. Private repo — examiner access only.
    """
    from datasets import Dataset, Features, Image, Value
    from huggingface_hub import HfApi

    print(f"\n=== Pushing full private → {repo_id} ===")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    # 1. Reasoning JSONLs as Parquet — same as the public repo, 4 configs/splits.
    print("\n[reasoning configs]")
    for fmt in ("full_reasoning", "label_only"):
        from datasets import DatasetDict
        splits = {}
        for split in ("train", "val"):
            jp = DATA_FT / fmt / f"{split}.jsonl"
            rows = load_and_augment(jp, source_map)
            splits[split] = Dataset.from_list(rows)
            print(f"  {fmt}/{split}: {len(rows)} rows")
        DatasetDict(splits).push_to_hub(repo_id, config_name=fmt, private=True)

    # 2. Full image set under config_name="images" — every train image with class + source.
    print("\n[images config]")
    rows: list[dict[str, Any]] = []
    n_zero = 0
    for cls_dir in FINAL_TRAIN.iterdir():
        if not cls_dir.is_dir():
            continue
        for img in cls_dir.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if img.stat().st_size == 0:
                n_zero += 1
                continue
            rows.append({
                "image": str(img),
                "image_id": img.stem,
                "class": cls_dir.name,
                "source": attribute(str(img.relative_to(REPO_ROOT)), source_map),
            })
    print(f"  {len(rows)} images (skipped {n_zero} zero-byte)")
    features = Features({
        "image": Image(),
        "image_id": Value("string"),
        "class": Value("string"),
        "source": Value("string"),
    })
    img_ds = Dataset.from_list(rows, features=features)
    img_ds.push_to_hub(repo_id, config_name="images", private=True)
    print("  full private push complete")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["reasoning", "redistributable", "full"], required=True)
    p.add_argument("--repo-prefix", default="danielfdias98", help="Hub user/org prefix")
    p.add_argument("--dry-run", action="store_true", help="Build dataset locally but don't push")
    args = p.parse_args()

    print("Building source-attribution map from data/dataset/...")
    source_map = build_source_map()
    print(f"Indexed {len(source_map)} unique image basenames across {len(SOURCES)} sources.\n")

    if args.target == "reasoning":
        repo = f"{args.repo_prefix}/derm-reasoning"
        if args.dry_run:
            for fmt in ("full_reasoning", "label_only"):
                for split in ("train", "val"):
                    rows = load_and_augment(DATA_FT / fmt / f"{split}.jsonl", source_map)
                    print(f"[dry-run] {fmt}/{split}: {len(rows)} rows")
            return
        push_reasoning(repo, source_map)
    elif args.target == "redistributable":
        repo = f"{args.repo_prefix}/derm-reasoning-redistributable"
        if args.dry_run:
            print("[dry-run] would push CC-compatible image subset")
            return
        push_redistributable(repo, source_map)
    elif args.target == "full":
        repo = f"{args.repo_prefix}/derm-reasoning-full"
        if args.dry_run:
            print("[dry-run] would push full private repo")
            return
        push_full_private(repo, source_map)


if __name__ == "__main__":
    main()
