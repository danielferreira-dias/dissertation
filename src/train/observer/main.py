"""
Stage 1: Observer — Generate visual descriptions for dermatology images.

Describes images WITHOUT any diagnosis knowledge using Vertex AI Gemini.

Quick start (Gemini 3 Flash on Vertex AI, 8 parallel workers):
  python3 src/train/observer/main.py --model gemini-3-flash-preview --data-dir final/train --workers 8

Test with 5 images:
  python3 src/train/observer/main.py --model gemini-3-flash-preview --data-dir final/train --limit 1 --classes melanoma psoriasis eczema basal_cell_carcinoma seborrheic_keratosis

Retry previously flagged images:
  python3 src/train/observer/main.py --retry-flagged

Script is resumable — re-run the same command to continue from where it left off.

Setup:
  gcloud auth application-default login
  gcloud config set project YOUR_PROJECT_ID

Available models: gemini-3-flash-preview, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro

Output: data/reasoning/observations.jsonl
Flagged: data/reasoning/flagged.jsonl
"""

import argparse
import io
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

load_dotenv()

OBSERVER_PROMPT = """You are a clinical dermatology observer. Describe ONLY what you see in this image.

RULES:
- Do NOT suggest any diagnosis or condition name.
- Do NOT speculate about what the condition might be.
- Ignore watermarks, logos, or text overlays.
- Adapt descriptions to the patient's skin tone (use "hyperpigmented" or "violaceous" for darker skin instead of assuming "erythematous").
- If image quality is poor, describe what you can observe and note limitations.

Respond with ONLY a JSON object:
{
  "morphology": "primary lesion type and arrangement (1-2 sentences)",
  "color": "color description adapted to observed skin tone",
  "texture": "surface characteristics",
  "border": "border characteristics",
  "distribution": "anatomical location and pattern",
  "size_extent": "estimated size or extent",
  "additional_features": "other notable features, or empty string if none",
  "image_quality": "good | acceptable | poor",
  "estimated_fitzpatrick": 1-6
}

No markdown, no code fences, no diagnosis. Only the JSON object."""

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_RETRIES = 3


def collect_images(data_dir: Path, classes: list[str] | None) -> list[dict]:
    """Collect image paths from class directories."""
    images = []
    if classes:
        dirs = [data_dir / cls for cls in classes]
    else:
        dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    for cls_dir in dirs:
        if not cls_dir.exists():
            print(f"Warning: {cls_dir} not found, skipping.")
            continue
        label = cls_dir.name
        for f in sorted(cls_dir.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append({"path": str(f), "label": label})
    return images


def load_progress(output_path: Path) -> set[str]:
    """Load already-processed image paths."""
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done.add(entry["image_path"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def parse_response(text: str) -> dict | None:
    """Parse JSON response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def observe_image(client: genai.Client, image_path: str, model: str) -> tuple[dict | None, str | None]:
    """Send image to Vertex AI Gemini and get pure visual description.

    Retries up to MAX_RETRIES on transient failures.
    Returns (result, blocked_reason).
    """
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text=OBSERVER_PROMPT),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    safety_settings=SAFETY_SETTINGS,
                    response_mime_type="application/json",
                    max_output_tokens=1024,
                    temperature=0.2,
                ),
            )
        except Exception as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ["safety", "blocked", "recitation"]):
                return None, f"content_filter | {str(e)[:200]}"
            if any(kw in error_msg for kw in ["429", "503", "quota", "overloaded", "unavailable"]) and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise

        # Check if blocked by safety filters
        if response.candidates and response.candidates[0].finish_reason:
            reason = str(response.candidates[0].finish_reason)
            if reason in ("SAFETY", "RECITATION", "BLOCKED"):
                ratings = []
                if response.candidates[0].safety_ratings:
                    ratings = [f"{r.category}:{r.probability}" for r in response.candidates[0].safety_ratings]
                return None, f"{reason} | {', '.join(ratings)}"

        if not response.text:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return None, "empty_response"

        parsed = parse_response(response.text)
        if parsed is None:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return None, "json_parse_error"
        return parsed, None

    return None, "max_retries_exceeded"


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Observer — visual descriptions via Vertex AI Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model examples:
  --model gemini-3-flash-preview     Gemini 3 Flash (default)
  --model gemini-2.5-flash           Gemini 2.5 Flash
  --model gemini-2.5-flash-lite      Gemini 2.5 Flash Lite (cheapest)
  --model gemini-2.5-pro             Gemini 2.5 Pro (best quality)
        """,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("final/train"))
    parser.add_argument("--output", type=Path, default=Path("data/reasoning/observations.jsonl"))
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model name")
    parser.add_argument("--project", default=None, help="GCP project ID (default: from gcloud config)")
    parser.add_argument("--location", default="global", help="Vertex AI location (default: global)")
    parser.add_argument("--limit", type=int, default=None, help="Max images per class")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between API calls")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--retry-flagged", action="store_true", help="Retry previously flagged images")
    parser.add_argument("--classes", nargs="+", default=None, help="Specific classes (default: all)")
    args = parser.parse_args()

    print(f"Model: {args.model} | Workers: {args.workers} | Location: {args.location}")

    client = genai.Client(
        vertexai=True,
        project=args.project,
        location=args.location,
    )

    images = collect_images(args.data_dir, args.classes)
    print(f"Found {len(images)} images")

    if args.limit:
        limited = []
        class_counts = {}
        for img in images:
            count = class_counts.get(img["label"], 0)
            if count < args.limit:
                limited.append(img)
                class_counts[img["label"]] = count + 1
        images = limited
        print(f"Limited to {len(images)} images ({args.limit} per class)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    flagged_path = args.output.parent / "flagged.jsonl"

    done = load_progress(args.output)

    if args.retry_flagged and flagged_path.exists():
        flagged_paths = set()
        with open(flagged_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    flagged_paths.add(entry["image_path"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Retrying {len(flagged_paths)} previously flagged images")
        flagged_path.write_text("")
        remaining = [img for img in images if img["path"] not in done or img["path"] in flagged_paths]
    else:
        remaining = [img for img in images if img["path"] not in done]

    print(f"Already processed: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    failed = 0
    flagged_count = 0
    processed = 0
    lock = __import__("threading").Lock()

    def process_image(img: dict) -> None:
        nonlocal failed, flagged_count, processed
        try:
            result, blocked_reason = observe_image(client, img["path"], args.model)

            with lock:
                if blocked_reason:
                    flagged_count += 1
                    flagged_entry = {
                        "image_path": img["path"],
                        "label": img["label"],
                        "reason": blocked_reason,
                    }
                    flagged_out.write(json.dumps(flagged_entry) + "\n")
                    flagged_out.flush()
                else:
                    entry = {
                        "image_path": img["path"],
                        "label": img["label"],
                        "observation": result,
                    }
                    out.write(json.dumps(entry) + "\n")
                    out.flush()
                    processed += 1

        except Exception as e:
            with lock:
                failed += 1
                tqdm.write(f"Error: {img['path']}: {e}")

        time.sleep(args.delay)

    from concurrent.futures import ThreadPoolExecutor

    with open(args.output, "a") as out, open(flagged_path, "a") as flagged_out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(tqdm(
                pool.map(process_image, remaining),
                total=len(remaining),
                desc="Observing images",
            ))

    print(f"\nDone. Processed: {processed}, Flagged: {flagged_count}, Failed: {failed}")
    print(f"Output: {args.output}")
    if flagged_count > 0:
        print(f"Flagged images: {flagged_path}")


if __name__ == "__main__":
    main()
