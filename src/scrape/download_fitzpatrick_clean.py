"""
Clean Fitzpatrick17k Downloader

Downloads Fitzpatrick17k images filtered through CleanPatrick annotations
to remove off-topic samples, near-duplicates, and label errors.

Then extracts only the 6 target skin condition classes.
Note: Fitzpatrick17k has no "atopic dermatitis" label.

Prerequisites:
  pip install pandas requests datasets tqdm

Output structure:
  data/
    raw/fitzpatrick17k/
      fitzpatrick17k.csv          — original CSV from GitHub
      images/                     — all downloaded images
    processed/fitzpatrick_clean/  — filtered 6-class dataset
      melanoma/
      basal_cell_carcinoma/
      seborrheic_keratosis/
      psoriasis/
      eczema/
      seborrheic_dermatitis/
"""

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fitzpatrick17k"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset" / "fitzpatrick17k"

# Fitzpatrick17k CSV URL (from GitHub repo)
FITZ_CSV_URL = (
    "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/"
    "main/fitzpatrick17k.csv"
)

# Exact label -> target class mapping (from CleanPatrick's granular_partition_label).
# Fitzpatrick17k has NO "atopic dermatitis" label.
LABEL_TO_CLASS = {
    # Seborrheic Dermatitis
    "seborrheic dermatitis": "seborrheic_dermatitis",
    # Seborrheic Keratosis
    "seborrheic keratosis": "seborrheic_keratosis",
    # Basal Cell Carcinoma
    "basal cell carcinoma": "basal_cell_carcinoma",
    "basal cell carcinoma morpheiform": "basal_cell_carcinoma",
    "solid cystic basal cell carcinoma": "basal_cell_carcinoma",
    # Melanoma
    "melanoma": "melanoma",
    "superficial spreading melanoma ssm": "melanoma",
    "malignant melanoma": "melanoma",
    "lentigo maligna": "melanoma",
    # Psoriasis
    "psoriasis": "psoriasis",
    "pustular psoriasis": "psoriasis",
    # Eczema
    "eczema": "eczema",
    "dyshidrotic eczema": "eczema",
    # Acne
    "acne vulgaris": "acne",
    "acne": "acne",
    # Rosacea
    "rosacea": "rosacea",
    "rhinophyma": "rosacea",
    # Allergic Contact Dermatitis
    "allergic contact dermatitis": "contact_dermatitis",
    # Folliculitis
    "folliculitis": "folliculitis",
    # Urticaria
    "urticaria": "urticaria",
    "urticaria pigmentosa": "urticaria",
    # Scabies
    "scabies": "scabies",
    # Vitiligo
    "vitiligo": "vitiligo",
    # Actinic Keratosis
    "actinic keratosis": "actinic_keratosis",
    # Squamous Cell Carcinoma
    "squamous cell carcinoma": "squamous_cell_carcinoma",
    # Keratosis Pilaris
    "keratosis pilaris": "keratosis_pilaris",
    # Perioral Dermatitis
    "perioral dermatitis": "perioral_dermatitis",
    # Prurigo Nodularis
    "prurigo nodularis": "prurigo_nodularis",
    # Pityriasis Rosea
    "pityriasis rosea": "pityriasis_rosea",
    # Drug Eruption
    "drug eruption": "drug_eruption",
    "fixed eruptions": "drug_eruption",
    # Hidradenitis
    "hidradenitis": "hidradenitis",
    # Erythema Multiforme
    "erythema multiforme": "erythema_multiforme",
    # Granuloma Annulare
    "granuloma annulare": "granuloma_annulare",
    # Acanthosis Nigricans
    "acanthosis nigricans": "acanthosis_nigricans",
    # Lichen Planus
    "lichen planus": "lichen_planus",
    # Dermatofibroma
    "dermatofibroma": "dermatofibroma",
    # Kaposi Sarcoma
    "kaposi sarcoma": "kaposi_sarcoma",
    # Keloid
    "keloid": "keloid",
    # Lupus
    "lupus erythematosus": "lupus",
    "lupus subacute": "lupus",
    # Stasis Edema / Dermatitis
    "stasis edema": "stasis_dermatitis",
}

# Download settings
MAX_WORKERS = 8
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2


def download_csv() -> pd.DataFrame:
    """Download the Fitzpatrick17k CSV if not cached."""
    csv_path = RAW_DIR / "fitzpatrick17k.csv"

    if csv_path.exists():
        log.info("Using cached CSV: %s", csv_path)
        return pd.read_csv(csv_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Fitzpatrick17k CSV...")
    resp = requests.get(FITZ_CSV_URL, timeout=30)
    resp.raise_for_status()
    csv_path.write_bytes(resp.content)
    log.info("CSV saved to %s", csv_path)
    return pd.read_csv(csv_path)


def load_cleanpatrick() -> pd.DataFrame:
    """Load CleanPatrick combined annotations from HuggingFace."""
    log.info("Loading CleanPatrick annotations from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "Digital-Dermatology/CleanPatrick",
        "Combined Annotations",
        split="test",
    )
    df = ds.to_pandas()
    log.info(
        "CleanPatrick loaded: %d images, columns: %s",
        len(df), list(df.columns),
    )
    return df


def filter_clean_images(fitz_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Filter Fitzpatrick17k using CleanPatrick annotations.

    Removes:
      - off-topic samples (off_topic_sample == 1)
      - label errors (label_error == 1)
      - near-duplicates (keep first per nd_clique group)
    """
    # Fitzpatrick17k filenames are {md5hash}.jpg — extract the hash
    # The `md5hash` column in Fitzpatrick CSV might not exist, so we
    # derive it from the URL or match on the `md5hash` from CleanPatrick.
    total = len(fitz_df)

    # Merge on md5hash
    if "md5hash" not in fitz_df.columns:
        # Derive md5hash from the image URL (last path segment without extension)
        fitz_df["md5hash"] = fitz_df["url"].apply(
            lambda u: Path(str(u)).stem if pd.notna(u) else ""
        )

    merged = fitz_df.merge(clean_df, on="md5hash", how="inner")
    log.info("Matched %d / %d images with CleanPatrick annotations", len(merged), total)

    # Step 1: Remove off-topic
    before = len(merged)
    merged = merged[merged["off_topic_sample"] == 0]
    log.info("Removed %d off-topic samples", before - len(merged))

    # Step 2: Remove label errors
    before = len(merged)
    merged = merged[merged["label_error"] == 0]
    log.info("Removed %d label errors", before - len(merged))

    # Step 3: Deduplicate — keep first image per near-duplicate clique
    before = len(merged)
    merged = merged.drop_duplicates(subset="nd_clique", keep="first")
    log.info("Removed %d near-duplicates", before - len(merged))

    log.info("Clean images remaining: %d / %d (%.1f%%)", len(merged), total, len(merged) / total * 100)
    return merged


def map_to_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Map Fitzpatrick17k labels to target classes using exact matching."""
    # Use `label` from Fitzpatrick17k CSV (fine-grained diagnosis)
    label_col = "label" if "label" in df.columns else "granular_partition_label"

    df = df.copy()
    df["target_class"] = df[label_col].str.lower().map(LABEL_TO_CLASS)

    # Keep only matched rows
    matched = df[df["target_class"].notna()]
    log.info(
        "Mapped %d images to 6 target classes (from %d clean images)",
        len(matched), len(df),
    )

    # Show per-class counts
    counts = matched["target_class"].value_counts()
    for cls, count in counts.items():
        log.info("  %s: %d images", cls, count)

    return matched


def download_image(url: str, dest: Path, retries: int = MAX_RETRIES) -> bool:
    """Download a single image. Returns True on success."""
    if dest.exists():
        return True

    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()

            # Verify it's actually an image
            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type and "octet-stream" not in content_type:
                return False

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True

        except (requests.RequestException, IOError):
            if attempt < retries:
                time.sleep(1 * (attempt + 1))
            continue

    return False


def download_and_organise(df: pd.DataFrame):
    """Download images and organise into class folders."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = RAW_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    total = len(df)
    success = 0
    failed = 0
    skipped = 0

    log.info("Downloading %d images with %d workers...", total, MAX_WORKERS)

    def process_row(row):
        url = row["url"]
        md5 = row["md5hash"]
        cls = row["target_class"]

        if pd.isna(url) or not str(url).startswith("http"):
            return "skip", cls

        # Download to raw cache
        ext = Path(str(url)).suffix or ".jpg"
        raw_path = images_dir / f"{md5}{ext}"
        dest_path = OUTPUT_DIR / cls / f"{md5}{ext}"

        if dest_path.exists():
            return "exists", cls

        if download_image(str(url), raw_path):
            # Copy to class folder
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_path, dest_path)
            return "ok", cls
        else:
            return "fail", cls

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_row, row): idx
            for idx, row in df.iterrows()
        }

        with tqdm(total=total, desc="Downloading") as pbar:
            for future in as_completed(futures):
                result, cls = future.result()
                if result in ("ok", "exists"):
                    success += 1
                elif result == "fail":
                    failed += 1
                else:
                    skipped += 1
                pbar.update(1)

    log.info("Download complete: %d success, %d failed, %d skipped", success, failed, skipped)


def print_summary():
    """Print final dataset summary."""
    print("\n" + "=" * 55)
    print("CLEAN FITZPATRICK17K DATASET SUMMARY")
    print("=" * 55)
    total = 0
    for class_dir in sorted(OUTPUT_DIR.iterdir()):
        if class_dir.is_dir():
            n = len([f for f in class_dir.iterdir() if f.is_file()])
            total += n
            print(f"  {class_dir.name:30s} {n:>5d} images")
    print(f"  {'TOTAL':30s} {total:>5d} images")
    print(f"\n  Output: {OUTPUT_DIR}")


def main():
    log.info("=" * 55)
    log.info("Clean Fitzpatrick17k Downloader")
    log.info("=" * 55)

    # Step 1: Load Fitzpatrick17k CSV
    log.info("\n--- Step 1: Load Fitzpatrick17k CSV ---")
    fitz_df = download_csv()
    log.info("Fitzpatrick17k: %d images, columns: %s", len(fitz_df), list(fitz_df.columns))

    # Step 2: Load CleanPatrick annotations
    log.info("\n--- Step 2: Load CleanPatrick annotations ---")
    clean_df = load_cleanpatrick()

    # Step 3: Filter out bad images
    log.info("\n--- Step 3: Filter using CleanPatrick ---")
    clean_fitz = filter_clean_images(fitz_df, clean_df)

    # Step 4: Map to 6 target classes
    log.info("\n--- Step 4: Map to target classes ---")
    target_df = map_to_classes(clean_fitz)

    if target_df.empty:
        log.error("No images matched target classes. Check LABEL_TO_CLASS mapping.")
        return

    # Step 5: Download and organise
    log.info("\n--- Step 5: Download and organise images ---")
    download_and_organise(target_df)

    # Summary
    print_summary()


if __name__ == "__main__":
    main()
