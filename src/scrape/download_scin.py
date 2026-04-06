"""
SCIN (Skin Condition Image Network) Downloader

Downloads clinical smartphone photos from Google's SCIN dataset,
filtered to target skin condition classes.

SCIN contains ~10,000+ crowd-sourced images with dermatologist labels.
Images are real smartphone photos (not dermoscopic), ideal for clinical use.

Data is stored on Google Cloud Storage: gs://dx-scin-public-data/

Prerequisites:
  pip install pandas requests tqdm

Output structure:
  data/
    raw/scin/
      scin_cases.csv
      scin_labels.csv
      images/                       — all downloaded target images
    processed/scin_clean/           — organised by class
      melanoma/
      basal_cell_carcinoma/
      seborrheic_keratosis/
      psoriasis/
      eczema/
      atopic_dermatitis/
      seborrheic_dermatitis/
"""

import ast
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
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "scin"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset" / "scin"

GCS_BUCKET = "https://storage.googleapis.com/dx-scin-public-data"
CASES_URL = f"{GCS_BUCKET}/dataset/scin_cases.csv"
LABELS_URL = f"{GCS_BUCKET}/dataset/scin_labels.csv"

# Map SCIN weighted_skin_condition_label keys to our target classes.
# SCIN uses broad condition names in the weighted label dict.
LABEL_TO_CLASS = {
    # Seborrheic Dermatitis
    "Seborrheic Dermatitis": "seborrheic_dermatitis",
    # Eczema
    "Eczema": "eczema",
    "Acute and chronic dermatitis": "eczema",
    "Acute constitutional eczema": "eczema",
    "Infected eczema": "eczema",
    "Lichenified eczematous dermatitis": "eczema",
    "Lichenified eczema": "eczema",
    "Acute-on-chronic dyshidrotic eczema of hands": "eczema",
    "Crusted eczematous dermatitis": "eczema",
    "Stasis Dermatitis": "eczema",
    # Atopic Dermatitis
    "Chronic dermatitis, NOS": "atopic_dermatitis",
    "Acute dermatitis, NOS": "atopic_dermatitis",
    "Lichen Simplex Chronicus": "atopic_dermatitis",
    # Psoriasis
    "Psoriasis": "psoriasis",
    "Inverse psoriasis": "psoriasis",
    "Psoriasiform dermatitis": "psoriasis",
    # Basal Cell Carcinoma
    "Basal Cell Carcinoma": "basal_cell_carcinoma",
    # Melanoma
    "Melanoma": "melanoma",
    # Seborrheic Keratosis
    "SK/ISK": "seborrheic_keratosis",
    # Acne / Rosacea
    "Acne": "acne",
    "Rosacea": "acne",
    # Tinea / Ringworm
    "Tinea": "tinea",
    "Tinea Versicolor": "tinea",
    "Onychomycosis": "tinea",
    "Deep fungal infection": "tinea",
    # Contact Dermatitis
    "Allergic Contact Dermatitis": "contact_dermatitis",
    "Irritant Contact Dermatitis": "contact_dermatitis",
    "CD - Contact dermatitis": "contact_dermatitis",
    "Contact dermatitis, NOS": "contact_dermatitis",
    # Urticaria
    "Urticaria": "urticaria",
    # Folliculitis
    "Folliculitis": "folliculitis",
    # Insect Bite
    "Insect Bite": "insect_bite",
    # Impetigo
    "Impetigo": "impetigo",
    # Herpes Zoster
    "Herpes Zoster": "herpes_zoster",
    # Herpes Simplex
    "Herpes Simplex": "herpes_simplex",
    # Drug Rash
    "Drug Rash": "drug_rash",
    # Pityriasis Rosea
    "Pityriasis rosea": "pityriasis_rosea",
    # Pigmented Purpuric Eruption
    "Pigmented purpuric eruption": "pigmented_purpuric_eruption",
    # Keratosis Pilaris
    "Keratosis pilaris": "keratosis_pilaris",
    # Scabies
    "Scabies": "scabies",
    # Actinic Keratosis
    "Actinic Keratosis": "actinic_keratosis",
}

# Minimum confidence threshold for the weighted label
MIN_CONFIDENCE = 0.3

# Download settings
MAX_WORKERS = 8
REQUEST_TIMEOUT = 20
MAX_RETRIES = 2


def download_csv(url: str, dest: Path) -> pd.DataFrame:
    """Download a CSV from GCS if not cached."""
    if dest.exists():
        log.info("Using cached: %s", dest)
        return pd.read_csv(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s ...", url.split("/")[-1])
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    log.info("Saved to %s", dest)
    return pd.read_csv(dest)


def parse_weighted_label(label_str: str) -> dict:
    """Parse the weighted_skin_condition_label string into a dict."""
    if pd.isna(label_str) or not label_str.strip():
        return {}
    try:
        return ast.literal_eval(label_str)
    except (ValueError, SyntaxError):
        return {}


def assign_class(weighted_label: dict) -> tuple[str | None, float]:
    """Assign a target class based on the weighted label dict.

    Returns (class_name, confidence) or (None, 0.0) if no match.
    Uses the highest-confidence matching condition.
    """
    best_class = None
    best_conf = 0.0

    for condition, weight in weighted_label.items():
        if condition in LABEL_TO_CLASS and weight > best_conf:
            best_class = LABEL_TO_CLASS[condition]
            best_conf = weight

    if best_conf >= MIN_CONFIDENCE:
        return best_class, best_conf
    return None, 0.0


def build_image_list(cases_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Join cases with labels and build a list of images to download."""
    # Merge on case_id
    merged = cases_df.merge(labels_df[["case_id", "weighted_skin_condition_label"]], on="case_id", how="inner")
    log.info("Merged %d cases with labels", len(merged))

    # Parse weighted labels and assign classes
    merged["parsed_label"] = merged["weighted_skin_condition_label"].apply(parse_weighted_label)
    results = merged["parsed_label"].apply(assign_class)
    merged["target_class"] = results.apply(lambda x: x[0])
    merged["confidence"] = results.apply(lambda x: x[1])

    # Filter to matched classes only
    matched = merged[merged["target_class"].notna()].copy()
    log.info("Matched %d cases to target classes", len(matched))

    # Explode image paths (up to 3 per case)
    image_rows = []
    for _, row in matched.iterrows():
        for i in range(1, 4):
            path_col = f"image_{i}_path"
            if path_col in row and pd.notna(row[path_col]) and str(row[path_col]).strip():
                image_rows.append({
                    "case_id": row["case_id"],
                    "image_path": row[path_col],
                    "target_class": row["target_class"],
                    "confidence": row["confidence"],
                })

    images_df = pd.DataFrame(image_rows)
    log.info("Total images to download: %d", len(images_df))

    # Show per-class counts
    counts = images_df["target_class"].value_counts()
    for cls, count in counts.items():
        log.info("  %s: %d images", cls, count)

    return images_df


def download_image(gcs_path: str, dest: Path, retries: int = MAX_RETRIES) -> bool:
    """Download a single image from GCS. Returns True on success."""
    if dest.exists():
        return True

    url = f"{GCS_BUCKET}/{gcs_path}"
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()

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


def download_and_organise(images_df: pd.DataFrame):
    """Download images and organise into class folders."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_images_dir = RAW_DIR / "images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    total = len(images_df)
    success = 0
    failed = 0

    log.info("Downloading %d images with %d workers...", total, MAX_WORKERS)

    def process_row(row):
        gcs_path = row["image_path"]
        cls = row["target_class"]
        filename = Path(gcs_path).name

        # Download to raw cache
        raw_path = raw_images_dir / filename
        dest_path = OUTPUT_DIR / cls / filename

        if dest_path.exists():
            return "exists"

        if download_image(gcs_path, raw_path):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_path, dest_path)
            return "ok"
        else:
            return "fail"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_row, row): idx
            for idx, row in images_df.iterrows()
        }

        with tqdm(total=total, desc="Downloading") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result in ("ok", "exists"):
                    success += 1
                else:
                    failed += 1
                pbar.update(1)

    log.info("Download complete: %d success, %d failed", success, failed)


def print_summary():
    """Print final dataset summary."""
    print("\n" + "=" * 55)
    print("SCIN CLEAN DATASET SUMMARY")
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
    log.info("SCIN Dataset Downloader")
    log.info("=" * 55)

    # Step 1: Download CSVs
    log.info("\n--- Step 1: Download metadata CSVs ---")
    cases_df = download_csv(CASES_URL, RAW_DIR / "scin_cases.csv")
    labels_df = download_csv(LABELS_URL, RAW_DIR / "scin_labels.csv")
    log.info("Cases: %d rows, Labels: %d rows", len(cases_df), len(labels_df))

    # Step 2: Build image list
    log.info("\n--- Step 2: Match to target classes ---")
    images_df = build_image_list(cases_df, labels_df)

    if images_df.empty:
        log.error("No images matched target classes.")
        return

    # Step 3: Download and organise
    log.info("\n--- Step 3: Download and organise images ---")
    download_and_organise(images_df)

    # Summary
    print_summary()


if __name__ == "__main__":
    main()
