"""
Dataset Downloader

Downloads and organises the 6-class skin disease dataset from two sources:
  - Kaggle DermNet (shubhamgoel27/dermnet) — 5 classes
  - SkinDisNet (Mendeley Data) — Seborrheic Dermatitis class

Prerequisites:
  pip install kaggle requests python-dotenv
  # Set KAGGLE_API_TOKEN in .env at the project root

Output structure:
  data/
    raw/kaggle_dermnet/          — full Kaggle download (23 classes)
    raw/skindisnet/              — full SkinDisNet download (6 classes)
    processed/                   — final 6-class dataset
      melanoma/
      seborrheic_keratoses/
      psoriasis/
      eczema/
      atopic_dermatitis/
      seborrheic_dermatitis/
"""

import logging
import os
import shutil
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env from project root (contains KAGGLE_API_TOKEN)
load_dotenv(PROJECT_ROOT / ".env")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Kaggle class folder names -> our class labels
KAGGLE_CLASS_MAP = {
    "Melanoma Skin Cancer Nevi and Moles": "melanoma",
    "Seborrheic Keratoses and other Benign Tumors": "seborrheic_keratoses",
    "Psoriasis pictures Lichen Planus and related diseases": "psoriasis",
    "Eczema Photos": "eczema",
    "Atopic Dermatitis Photos": "atopic_dermatitis",
}

# SkinDisNet Mendeley download URL (version 2)
SKINDISNET_URL = "https://data.mendeley.com/public-api/zip/yj3md44hxg/download/2"
SKINDISNET_CLASS = "Seborrheic Dermatitis"
SKINDISNET_LABEL = "seborrheic_dermatitis"


def download_kaggle_dermnet():
    """Download the DermNet dataset from Kaggle using the kaggle CLI."""
    kaggle_dir = RAW_DIR / "kaggle_dermnet"

    if kaggle_dir.exists() and any(kaggle_dir.iterdir()):
        log.info("Kaggle DermNet already downloaded at %s", kaggle_dir)
        return kaggle_dir

    kaggle_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading DermNet from Kaggle (~1.8 GB)...")
    exit_code = os.system(
        f"kaggle datasets download -d shubhamgoel27/dermnet "
        f"-p {kaggle_dir} --unzip"
    )
    if exit_code != 0:
        raise RuntimeError(
            "Kaggle download failed. Make sure:\n"
            "  1. pip install kaggle\n"
            "  2. ~/.kaggle/kaggle.json exists (from kaggle.com > Settings > API)\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json"
        )

    log.info("Kaggle DermNet downloaded to %s", kaggle_dir)
    return kaggle_dir


def download_skindisnet():
    """Download the SkinDisNet dataset from Mendeley Data."""
    skindisnet_dir = RAW_DIR / "skindisnet"
    zip_path = skindisnet_dir / "skindisnet.zip"

    if skindisnet_dir.exists() and any(
        p for p in skindisnet_dir.iterdir() if p.is_dir()
    ):
        log.info("SkinDisNet already downloaded at %s", skindisnet_dir)
        return skindisnet_dir

    skindisnet_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading SkinDisNet from Mendeley Data...")
    resp = requests.get(SKINDISNET_URL, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                print(f"\r  Progress: {pct}% ({downloaded // 1024 // 1024} MB)", end="")
    print()

    log.info("Extracting SkinDisNet...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(skindisnet_dir)
    zip_path.unlink()

    log.info("SkinDisNet downloaded to %s", skindisnet_dir)
    return skindisnet_dir


def find_kaggle_class_dirs(kaggle_dir: Path) -> dict[str, Path]:
    """Find the actual folder paths for each Kaggle class we want.

    The Kaggle dataset has train/ and test/ subdirectories, each containing
    class folders. Folder names may vary slightly, so we do fuzzy matching.
    """
    found = {}

    for split_dir in kaggle_dir.rglob("*"):
        if not split_dir.is_dir():
            continue
        folder_name = split_dir.name
        for kaggle_name, our_label in KAGGLE_CLASS_MAP.items():
            # Fuzzy match: check if key words from our expected name appear
            key_words = kaggle_name.lower().split()[:2]
            if all(w in folder_name.lower() for w in key_words):
                if our_label not in found:
                    found[our_label] = []
                found[our_label].append(split_dir)

    return found


def find_skindisnet_sd_dir(skindisnet_dir: Path) -> Path | None:
    """Find the Seborrheic Dermatitis folder in SkinDisNet."""
    for d in skindisnet_dir.rglob("*"):
        if d.is_dir() and "seborrheic" in d.name.lower():
            return d
    return None


def copy_images(src_dirs: list[Path], dst_dir: Path) -> int:
    """Copy all image files from source dirs to destination. Returns count."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    seen = set()

    for src_dir in src_dirs:
        for img in src_dir.iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                # Avoid name collisions across train/test splits
                name = img.name
                if name in seen:
                    stem = img.stem
                    name = f"{stem}_{count}{img.suffix}"
                seen.add(name)

                shutil.copy2(img, dst_dir / name)
                count += 1

    return count


def assemble_dataset():
    """Download sources and assemble the final 6-class dataset."""
    log.info("=" * 50)
    log.info("Step 1: Download Kaggle DermNet")
    log.info("=" * 50)
    kaggle_dir = download_kaggle_dermnet()

    log.info("=" * 50)
    log.info("Step 2: Download SkinDisNet")
    log.info("=" * 50)
    skindisnet_dir = download_skindisnet()

    log.info("=" * 50)
    log.info("Step 3: Assemble 6-class dataset")
    log.info("=" * 50)

    # Find Kaggle class directories
    kaggle_classes = find_kaggle_class_dirs(kaggle_dir)
    log.info("Found Kaggle classes: %s", list(kaggle_classes.keys()))

    # Copy Kaggle classes
    for label, src_dirs in kaggle_classes.items():
        dst = PROCESSED_DIR / label
        n = copy_images(src_dirs, dst)
        log.info("  %s: %d images (from %d dirs)", label, n, len(src_dirs))

    # Copy SkinDisNet Seborrheic Dermatitis
    sd_dir = find_skindisnet_sd_dir(skindisnet_dir)
    if sd_dir:
        dst = PROCESSED_DIR / SKINDISNET_LABEL
        n = copy_images([sd_dir], dst)
        log.info("  %s: %d images (from SkinDisNet)", SKINDISNET_LABEL, n)
    else:
        log.warning("Could not find Seborrheic Dermatitis folder in SkinDisNet!")
        log.warning("Available dirs: %s", [d.name for d in skindisnet_dir.rglob("*") if d.is_dir()])

    # Also merge DermNet NZ scraped images if they exist
    scraped_sd = PROJECT_ROOT / "data" / "dermnet_scraped" / "seborrheic_dermatitis"
    if scraped_sd.exists():
        dst = PROCESSED_DIR / SKINDISNET_LABEL
        n = copy_images([scraped_sd], dst)
        log.info("  %s: +%d images (from DermNet NZ scrape)", SKINDISNET_LABEL, n)

    # Print summary
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    total = 0
    for class_dir in sorted(PROCESSED_DIR.iterdir()):
        if class_dir.is_dir():
            n = len([f for f in class_dir.iterdir() if f.is_file()])
            total += n
            print(f"  {class_dir.name:30s} {n:>5d} images")
    print(f"  {'TOTAL':30s} {total:>5d} images")
    print(f"\n  Output: {PROCESSED_DIR}")


if __name__ == "__main__":
    assemble_dataset()
