"""
Stage 1: Observer — Generate visual descriptions for dermatology images.

Uses Gemini 3 Flash to describe images WITHOUT any diagnosis knowledge.
The observer sees only the image, producing pure visual observations.

Usage:
  python src/train/observer/main.py --data-dir final/train --output data/reasoning/observations.jsonl
  python src/train/observer/main.py --data-dir final/train --limit 5 --classes melanoma psoriasis
"""

import argparse
import io
import json
import os
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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


SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
]


def observe_image(client: genai.Client, image_path: str, model: str) -> tuple[dict | None, str | None]:
    """Send image to Gemini and get pure visual description.

    Returns (result, blocked_reason). If blocked, result is None and
    blocked_reason describes why.
    """
    img = Image.open(image_path).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

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
        config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS),
    )

    # Check if blocked by safety filters
    if response.candidates and response.candidates[0].finish_reason:
        reason = str(response.candidates[0].finish_reason)
        if reason in ("SAFETY", "RECITATION", "BLOCKED"):
            ratings = []
            if response.candidates[0].safety_ratings:
                ratings = [
                    f"{r.category}:{r.probability}" for r in response.candidates[0].safety_ratings
                ]
            return None, f"{reason} | {', '.join(ratings)}"

    if not response.text:
        return None, "empty_response"

    parsed = parse_response(response.text)
    if parsed is None:
        return None, "json_parse_error"
    return parsed, None


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Observer — visual descriptions")
    parser.add_argument("--data-dir", type=Path, default=Path("final/train"))
    parser.add_argument("--output", type=Path, default=Path("data/reasoning/observations.jsonl"))
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model (default: gemini-2.5-flash)")
    parser.add_argument("--limit", type=int, default=None, help="Max images per class")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--classes", nargs="+", default=None, help="Specific classes (default: all)")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_TOKEN")
    if not api_key:
        print("Error: GEMINI_API_TOKEN not set. Add it to .env or export it.")
        return

    client = genai.Client(api_key=api_key)

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
    done = load_progress(args.output)
    remaining = [img for img in images if img["path"] not in done]
    print(f"Already processed: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    flagged_path = args.output.parent / "flagged.jsonl"
    failed = 0
    flagged = 0
    with open(args.output, "a") as out, open(flagged_path, "a") as flagged_out:
        for img in tqdm(remaining, desc="Observing images"):
            try:
                result, blocked_reason = observe_image(client, img["path"], args.model)

                if blocked_reason:
                    flagged += 1
                    flagged_entry = {
                        "image_path": img["path"],
                        "label": img["label"],
                        "reason": blocked_reason,
                    }
                    flagged_out.write(json.dumps(flagged_entry) + "\n")
                    flagged_out.flush()
                    print(f"\nFlagged: {img['path']} — {blocked_reason}")
                    continue

                entry = {
                    "image_path": img["path"],
                    "label": img["label"],
                    "observation": result,
                }
                out.write(json.dumps(entry) + "\n")
                out.flush()

            except Exception as e:
                print(f"\nError: {img['path']}: {e}")
                failed += 1

            time.sleep(args.delay)

    print(f"\nDone. Processed: {len(remaining) - failed - flagged}, Flagged: {flagged}, Failed: {failed}")
    print(f"Output: {args.output}")
    if flagged > 0:
        print(f"Flagged images: {flagged_path}")


if __name__ == "__main__":
    main()
