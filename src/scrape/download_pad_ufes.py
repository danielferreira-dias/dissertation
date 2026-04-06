"""
PAD-UFES-20 Dataset Downloader

Downloads smartphone clinical images from the PAD-UFES-20 dataset (Mendeley Data).
Biopsy-proven labels for cancers. Brazilian patient population.

Source: https://data.mendeley.com/datasets/zr7vgbcyr2/1

Prerequisites:
  pip install pandas requests tqdm

Output:
  data/dataset/pad_ufes/
    basal_cell_carcinoma/
    melanoma/
    squamous_cell_carcinoma/
    actinic_keratosis/
    seborrheic_keratosis/
    nevus/
"""

import logging
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "pad_ufes"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset" / "pad_ufes"

# Mendeley download URL for the dataset
MENDELEY_URL = "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/f964a4e5-82b4-4a03-aab9-e64e4e84a77f/file_downloaded"

# Diagnostic label -> folder name mapping
LABEL_TO_CLASS = {
    "BCC": "basal_cell_carcinoma",
    "MEL": "melanoma",
    "SCC": "squamous_cell_carcinoma",
    "ACK": "actinic_keratosis",
    "SEK": "seborrheic_keratosis",
    "NEV": "nevus",
}


def download_dataset():
    """Download the ZIP from Mendeley if not cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "pad_ufes.zip"

    if zip_path.exists():
        log.info("Using cached ZIP: %s", zip_path)
        return zip_path

    log.info("Downloading PAD-UFES-20 from Mendeley...")
    resp = requests.get(MENDELEY_URL, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    log.info("Saved to %s", zip_path)
    return zip_path


def extract_and_organize(zip_path: Path):
    """Extract ZIP and organize images by class using metadata CSV."""
    extract_dir = RAW_DIR / "extracted"

    if not extract_dir.exists():
        log.info("Extracting ZIP...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        log.info("Extracted to %s", extract_dir)

    # Find metadata CSV
    csv_files = list(extract_dir.rglob("*.csv"))
    log.info("Found CSV files: %s", [str(f.name) for f in csv_files])

    metadata = None
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if "diagnostic" in df.columns.str.lower() or "img_id" in df.columns.str.lower():
                metadata = df
                log.info("Using metadata: %s (%d rows)", csv_file.name, len(df))
                break
        except Exception:
            continue

    # Find all images
    image_files = {}
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for f in extract_dir.rglob(ext):
            image_files[f.stem] = f

    log.info("Found %d images in extracted directory", len(image_files))

    if metadata is not None:
        organize_with_metadata(metadata, image_files)
    else:
        organize_by_folder(extract_dir)


def organize_with_metadata(df: pd.DataFrame, image_files: dict):
    """Organize using metadata CSV."""
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Find the diagnostic and image ID columns
    diag_col = next((c for c in df.columns if "diagnostic" in c), None)
    img_col = next((c for c in df.columns if "img_id" in c), None)

    if not diag_col:
        log.error("No diagnostic column found. Columns: %s", list(df.columns))
        return

    log.info("Using diagnostic column: '%s', image column: '%s'", diag_col, img_col)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Organizing"):
        diag = str(row[diag_col]).strip().upper()
        target_class = LABEL_TO_CLASS.get(diag)

        if not target_class:
            continue

        # Try to find the image
        img_id = str(row[img_col]).strip() if img_col else None
        src_path = image_files.get(img_id) if img_id else None

        if not src_path:
            continue

        dest_dir = OUTPUT_DIR / target_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name

        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)

        counts[target_class] = counts.get(target_class, 0) + 1

    print_summary(counts)


def organize_by_folder(extract_dir: Path):
    """Fallback: organize by scanning folder names if no metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}

    for img in extract_dir.rglob("*"):
        if not img.is_file() or img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        # Try to determine class from parent folder name
        parent = img.parent.name.upper()
        target_class = LABEL_TO_CLASS.get(parent)

        if not target_class:
            # Try matching partial folder names
            for abbrev, cls in LABEL_TO_CLASS.items():
                if abbrev in parent:
                    target_class = cls
                    break

        if target_class:
            dest_dir = OUTPUT_DIR / target_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / img.name
            if not dest_path.exists():
                shutil.copy2(img, dest_path)
            counts[target_class] = counts.get(target_class, 0) + 1

    print_summary(counts)


def print_summary(counts: dict):
    """Print and write summary."""
    print("\n" + "=" * 55)
    print("PAD-UFES-20 DATASET SUMMARY")
    print("=" * 55)
    total = 0
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:30s} {count:>5d} images")
        total += count
    print(f"  {'TOTAL':30s} {total:>5d} images")
    print(f"\n  Output: {OUTPUT_DIR}")

    # Write README
    readme = OUTPUT_DIR / "README.md"
    with open(readme, "w") as f:
        f.write("# PAD-UFES-20 Dataset\n\n")
        f.write("- **Source:** Mendeley Data (zr7vgbcyr2/1)\n")
        f.write("- **Type:** Smartphone clinical photos\n")
        f.write("- **Labels:** Biopsy-proven for cancers, clinical diagnosis for others\n")
        f.write("- **Population:** Brazilian patients\n")
        f.write("- **Metadata:** Fitzpatrick skin type, age, location, diameter\n")
        f.write("- **License:** CC BY 4.0\n\n")
        f.write("## Classes\n\n")
        f.write("| Class | Images |\n")
        f.write("|-------|--------|\n")
        for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
            f.write(f"| {cls} | {count:,} |\n")
        f.write(f"| **TOTAL** | **{total:,}** |\n")


def main():
    log.info("=" * 55)
    log.info("PAD-UFES-20 Dataset Downloader")
    log.info("=" * 55)

    zip_path = download_dataset()
    extract_and_organize(zip_path)


if __name__ == "__main__":
    main()
