"""
Derm1M Selective Extractor

Filters the Derm1M dataset for our 6 target conditions and extracts
only those images from the source zip files. Saves a filtered CSV
alongside the extracted images.

Usage:
    python src/scrape/extract_derm1m.py

Output:
    data/derm1m_filtered/
        images/              — extracted images (flat, renamed to avoid collisions)
        derm1m_filtered.csv  — filtered metadata with updated filenames
"""

import logging
import zipfile
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DERM1M_DIR = PROJECT_ROOT / "data" / "raw" / "derm1m"
OUTPUT_DIR = PROJECT_ROOT / "data" / "derm1m_filtered"
IMAGES_DIR = OUTPUT_DIR / "images"

# Our 6 target conditions and the search terms to match in disease_label.
# Each key is our class label; the value is a list of substrings to match.
TARGET_CONDITIONS = {
    "melanoma": ["melanoma"],
    "seborrheic_keratoses": ["seborrheic keratosis"],
    "basal_cell_carcinoma": ["basal cell carcinoma"],
    "psoriasis": ["psoriasis"],
    "eczema": ["eczema"],
    "atopic_dermatitis": ["atopic dermatitis"],
    "seborrheic_dermatitis": ["seborrheic dermatitis"],
}

# Map filename prefixes to zip files
ZIP_MAP = {
    "youtube/": "youtube.zip",
    "edu/": "edu.zip",
    "public/": "public.zip",
    "note/": "note.zip",
    "IIYI/": "IIYI.zip",
    "pubmed/": "pubmed.zip",
    "twitter/": "twitter.zip",
    "reddit/": "reddit.zip",
}


def classify_row(disease_label: str) -> str | None:
    """Return our class label if the disease_label matches a target condition."""
    if not disease_label:
        return None
    dl = disease_label.lower()
    for class_label, keywords in TARGET_CONDITIONS.items():
        if any(kw in dl for kw in keywords):
            return class_label
    return None


def get_zip_for_filename(filename: str) -> str | None:
    """Return the zip filename for a given image path."""
    for prefix, zipname in ZIP_MAP.items():
        if filename.startswith(prefix):
            return zipname
    return None


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DERM1M_DIR / "Derm1M_v2_pretrain.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Derm1M CSV not found at {csv_path}")

    # Step 1: Filter CSV for target conditions using pandas
    log.info("Step 1: Filtering CSV for target conditions...")
    df = pd.read_csv(csv_path, low_memory=False)
    total = len(df)

    df["our_label"] = df["disease_label"].apply(
        lambda x: classify_row(x) if isinstance(x, str) else None
    )
    filtered = df[df["our_label"].notna()].copy()
    log.info("Filtered %d / %d rows", len(filtered), total)

    for label, count in sorted(filtered["our_label"].value_counts().items()):
        log.info("  %s: %d", label, count)

    # Step 2: Group files by zip
    log.info("Step 2: Grouping files by source zip...")
    filtered["zipname"] = filtered["filename"].apply(get_zip_for_filename)
    skipped = filtered["zipname"].isna().sum()
    if skipped:
        log.warning("Skipped %d files with unknown zip mapping", skipped)

    zip_groups = filtered[filtered["zipname"].notna()].groupby("zipname")

    # Step 3: Extract images from each zip
    log.info("Step 3: Extracting images from zips...")
    extracted = 0
    failed = 0
    extracted_paths = {}

    for zipname, group_df in zip_groups:
        zip_path = DERM1M_DIR / zipname
        if not zip_path.exists():
            log.warning("Zip not found: %s (skipping %d files)", zipname, len(group_df))
            failed += len(group_df)
            continue

        # CSV filenames have prefix like "youtube/xxx.jpg" but zips contain
        # just "xxx.jpg" (no prefix). Strip the prefix to match zip contents.
        lookup = {}
        for csv_filename, label in zip(group_df["filename"], group_df["our_label"]):
            # Strip the source prefix (e.g. "youtube/" -> "")
            basename = csv_filename.split("/", 1)[-1] if "/" in csv_filename else csv_filename
            lookup[basename] = (csv_filename, label)

        log.info("  Opening %s (%d target files)...", zipname, len(lookup))
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = set(zf.namelist())
            for zip_name, (csv_filename, label) in lookup.items():
                if zip_name in namelist:
                    safe_name = csv_filename.replace("/", "_")
                    out_path = IMAGES_DIR / label / safe_name
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    if not out_path.exists():
                        data = zf.read(zip_name)
                        out_path.write_bytes(data)

                    extracted_paths[csv_filename] = str(out_path.relative_to(OUTPUT_DIR))
                    extracted += 1
                else:
                    failed += 1

        log.info("    Extracted so far: %d", extracted)

    log.info("Total extracted: %d, failed: %d", extracted, failed)

    # Step 4: Write filtered CSV
    log.info("Step 4: Writing filtered CSV...")
    filtered["extracted_path"] = filtered["filename"].map(extracted_paths)
    out_csv = OUTPUT_DIR / "derm1m_filtered.csv"
    filtered.to_csv(out_csv, index=False)

    # Summary
    print("\n" + "=" * 50)
    print("DERM1M EXTRACTION SUMMARY")
    print("=" * 50)
    total_extracted = 0
    for class_dir in sorted(IMAGES_DIR.iterdir()):
        if class_dir.is_dir():
            n = len(list(class_dir.iterdir()))
            total_extracted += n
            print(f"  {class_dir.name:30s} {n:>6d} images")
    print(f"  {'TOTAL':30s} {total_extracted:>6d} images")
    print(f"\n  Images: {IMAGES_DIR}")
    print(f"  CSV:    {out_csv}")


if __name__ == "__main__":
    main()
