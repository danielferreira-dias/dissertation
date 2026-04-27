"""Download the source images for danielfdias98/derm-reasoning.

This dataset publishes only the reasoning annotations (the JSONL/Parquet rows).
The 28k underlying images come from five separate datasets — most under
licenses that don't permit redistribution from this repo. Run this script to
re-create the image folder structure (`final/train/<class>/<file>`) that the
JSONL `image` column expects.

Sources you'll need access to:

  ✅ SCIN          — Google Cloud Storage public bucket  (no auth needed)
  ✅ PAD-UFES-20   — Kaggle                              (need kaggle.json)
  ⚠ SkinCAP       — HuggingFace                          (CC-BY-NC-SA 4.0)
  ⚠ Kaggle DermNet — Kaggle                              (need kaggle.json)
  ❌ DermNet NZ     — Not auto-downloadable (web-scraped). The script lists
                      the missing files; obtain manually with permission.

Usage:
    pip install datasets huggingface_hub kaggle google-cloud-storage
    python download_images.py --output ./final/train --sources scin,pad_ufes,skincap,kaggle_dermnet
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets first.", file=sys.stderr)
    sys.exit(1)


SOURCES = ("scin", "pad_ufes", "skincap", "kaggle_dermnet", "dermnet_nz")


def fetch_scin(out: Path, items: list[dict]) -> int:
    """Download SCIN images from Google Cloud Storage (Apache 2.0)."""
    try:
        from google.cloud import storage
    except ImportError:
        print("  pip install google-cloud-storage", file=sys.stderr)
        return 0
    # SCIN public bucket — bucket name and structure documented in the SCIN
    # release; image basenames in our `image_id` column match the GCS object
    # names. Implementation left to the user since the SCIN release format
    # has changed over time; see https://github.com/google-research-datasets/scin
    print("  TODO: implement SCIN GCS fetch — see https://github.com/google-research-datasets/scin")
    return 0


def fetch_pad_ufes(out: Path, items: list[dict]) -> int:
    """PAD-UFES-20 via Kaggle (CC BY 4.0)."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("  pip install kaggle && configure ~/.kaggle/kaggle.json")
        return 0
    import subprocess
    tmp = Path("./.cache/pad_ufes")
    tmp.mkdir(parents=True, exist_ok=True)
    if not (tmp / "metadata.csv").exists():
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "mahdavi1202/skin-cancer", "-p", str(tmp), "--unzip"],
            check=True,
        )
    # PAD-UFES-20 ships with metadata.csv mapping each `img_id` (PAT_xxx_xxx_xx)
    # to its diagnostic class. Walk and copy.
    n = 0
    for item in items:
        # Filename in our dataset matches PAD-UFES-20's img_id column verbatim.
        match = list(tmp.glob(f"**/{item['image_id']}.*"))
        if not match:
            continue
        dest = out / item["class"] / match[0].name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(match[0], dest)
        n += 1
    return n


def fetch_skincap(out: Path, items: list[dict]) -> int:
    """SkinCAP via HuggingFace (CC-BY-NC-SA 4.0)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  pip install huggingface_hub")
        return 0
    cache = snapshot_download(repo_id="joshuachou/SkinCAP", repo_type="dataset")
    cache = Path(cache)
    n = 0
    for item in items:
        match = list(cache.glob(f"**/{item['image_id']}.*"))
        if not match:
            continue
        dest = out / item["class"] / match[0].name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(match[0], dest)
        n += 1
    return n


def fetch_kaggle_dermnet(out: Path, items: list[dict]) -> int:
    """Kaggle DermNet — note: original images are scraped from DermNet, copyright applies."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("  pip install kaggle && configure ~/.kaggle/kaggle.json")
        return 0
    import subprocess
    tmp = Path("./.cache/kaggle_dermnet")
    tmp.mkdir(parents=True, exist_ok=True)
    if not any(tmp.iterdir()):
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "shubhamgoel27/dermnet", "-p", str(tmp), "--unzip"],
            check=True,
        )
    # Kaggle DermNet uses descriptive filenames like `basal-cell-carcinoma-face-1.jpg`.
    n = 0
    for item in items:
        match = list(tmp.glob(f"**/{item['image_id']}.*"))
        if not match:
            continue
        dest = out / item["class"] / match[0].name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(match[0], dest)
        n += 1
    return n


def fetch_dermnet_nz(out: Path, items: list[dict]) -> int:
    """DermNet NZ — not auto-downloadable. Print missing list."""
    print(f"  DermNet NZ: {len(items)} images. Not auto-downloadable; please obtain manually with permission.")
    print(f"  Sample missing image_ids:")
    for item in items[:5]:
        print(f"    {item['image_id']}  (class={item['class']})")
    print(f"  ... and {len(items) - 5} more.")
    return 0


HANDLERS = {
    "scin": fetch_scin,
    "pad_ufes": fetch_pad_ufes,
    "skincap": fetch_skincap,
    "kaggle_dermnet": fetch_kaggle_dermnet,
    "dermnet_nz": fetch_dermnet_nz,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("./final/train"),
                   help="Where to write final/train/<class>/<file>.{jpg,png}")
    p.add_argument("--sources", default=",".join(SOURCES),
                   help=f"Comma-separated subset of {SOURCES}")
    p.add_argument("--config", default="full_reasoning",
                   choices=("full_reasoning", "label_only"))
    p.add_argument("--split", default="train", choices=("train", "val"))
    args = p.parse_args()

    sources = [s.strip() for s in args.sources.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset metadata: danielfdias98/derm-reasoning [{args.config}/{args.split}]")
    ds = load_dataset("danielfdias98/derm-reasoning", args.config, split=args.split)
    print(f"  {len(ds)} rows")

    # Group by source
    by_src: dict[str, list[dict]] = {s: [] for s in sources}
    for row in ds:
        if row["source"] in by_src:
            by_src[row["source"]].append({"image_id": row["image_id"], "class": row["class"]})

    total = 0
    for src in sources:
        items = by_src.get(src, [])
        if not items:
            continue
        print(f"\n[{src}] {len(items)} rows")
        n = HANDLERS[src](args.output, items)
        print(f"  copied {n}")
        total += n

    print(f"\nTotal images copied: {total}")
    if total < len(ds):
        print(f"Missing: {len(ds) - total}. Check the per-source notes above.")


if __name__ == "__main__":
    main()
