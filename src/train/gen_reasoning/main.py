"""
Generate clinical reasoning descriptions for dermatology images using Gemini API.

Pipeline:
  1. Gemini generates reasoning anchored to expert labels → descriptions.jsonl
  2. convert_to_training.py transforms to chat format → train.jsonl
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

CATEGORY_MAP = {
    "melanoma": "malignant",
    "basal_cell_carcinoma": "malignant",
    "seborrheic_keratosis": "benign",
    "psoriasis": "inflammatory",
    "eczema": "inflammatory",
    "seborrheic_dermatitis": "inflammatory",
}

LABEL_DISPLAY = {
    "melanoma": "Melanoma",
    "basal_cell_carcinoma": "Basal Cell Carcinoma",
    "seborrheic_keratosis": "Seborrheic Keratosis",
    "psoriasis": "Psoriasis",
    "eczema": "Eczema",
    "seborrheic_dermatitis": "Seborrheic Dermatitis",
}

PROMPT_TEMPLATE = """You are a dermatology assistant analyzing clinical skin images for a research dataset.

INSTRUCTIONS:
- Ignore any watermarks, logos, or text overlays on the image. Focus only on the visible skin condition.
- Do NOT mention or reference any watermarks in your response.
- The confirmed expert diagnosis for this image is: {diagnosis} ({category}).
- Your job is to describe the visual features that support this diagnosis using separate structured attributes.
- If the image quality is poor or features are ambiguous, still describe what you observe and note the limitation.
- Adapt your description to the patient's skin tone. Avoid assuming light skin presentation (e.g., use "violaceous" or "hyperpigmented" instead of "erythematous" when appropriate for darker skin).
- IMPORTANT: If the image clearly shows a DIFFERENT condition than {diagnosis} (e.g., a keloid labeled as seborrheic keratosis), set "label_match" to false and explain in the reasoning field.

Respond ONLY with a valid JSON object in this exact format:

{{
  "label_match": true or false,
  "morphology": "Primary lesion type (plaque, papule, macule, nodule, vesicle, pustule, etc.) and arrangement (solitary, grouped, confluent). 1-2 sentences.",
  "color": "Exact color description adapted to the observed skin tone (e.g., violaceous, hyperpigmented, erythematous, depigmented, tan-brown). Note heterogeneity if present.",
  "texture": "Surface quality (smooth, rough, verrucous, scaly, crusted, lichenified, weeping, etc.).",
  "border": "Border characteristics (well-defined, irregular, asymmetric, diffuse, raised, flat).",
  "location": "Visible body location and distribution pattern (e.g., truncal, flexural, sun-exposed, dermatomal).",
  "size": "Estimated lesion size or extent if discernible (e.g., <1cm, 2-3cm, diffuse patches).",
  "reasoning": "1-2 sentences tying the above features to the confirmed diagnosis of {diagnosis}. If label_match is false, explain what the image actually shows and why it does not match {diagnosis}.",
  "confidence": "low | medium | high",
  "fitzpatrick_skin_type": 1-6
}}

No markdown, no code fences, no extra text. Only the JSON object."""

CLASSES = [
    "melanoma",
    "basal_cell_carcinoma",
    "psoriasis",
    "eczema",
    "seborrheic_keratosis",
    "seborrheic_dermatitis",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(data_dir: Path, classes: list[str]) -> list[dict]:
    """Collect all image paths and their labels from class directories."""
    images = []
    for cls in classes:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"Warning: {cls_dir} not found, skipping.")
            continue
        for f in sorted(cls_dir.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append({"path": str(f), "label": cls})
    return images


def load_progress(output_path: Path) -> set[str]:
    """Load already-processed image paths from existing output file."""
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
    """Parse Gemini response, handling markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def generate_description(
    client: genai.Client, image_path: str, label: str, model: str
) -> dict | None:
    """Send image to Gemini with expert label and parse the JSON response."""
    img = Image.open(image_path)
    img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    diagnosis = LABEL_DISPLAY[label]
    category = CATEGORY_MAP[label]
    prompt = PROMPT_TEMPLATE.format(diagnosis=diagnosis, category=category)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=prompt),
                ],
            )
        ],
    )

    if not response.text:
        return None

    return parse_response(response.text)


def main():
    parser = argparse.ArgumentParser(description="Generate clinical reasoning for derm images")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/dataset/train"),
        help="Directory containing class subdirectories with images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reasoning/descriptions.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images per class (None = all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=15.0,
        help="Seconds between API calls (default: 15 for free tier rate limit)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=CLASSES,
        help="Classes to process",
    )
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_TOKEN")
    if not api_key:
        print("Error: GEMINI_API_TOKEN not set. Add it to .env or export it.")
        return

    client = genai.Client(api_key=api_key)

    # Collect images
    images = collect_images(args.data_dir, args.classes)
    print(f"Found {len(images)} images across {len(args.classes)} classes")

    # Apply per-class limit if set
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

    # Resume from previous progress
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_progress(args.output)
    remaining = [img for img in images if img["path"] not in done]
    print(f"Already processed: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    # Process images
    failed = 0
    with open(args.output, "a") as out:
        for img in tqdm(remaining, desc="Generating descriptions"):
            try:
                result = generate_description(client, img["path"], img["label"], args.model)

                if result is None:
                    print(f"\nFailed to parse response for {img['path']}")
                    failed += 1
                    continue

                entry = {
                    "image_path": img["path"],
                    "label": img["label"],
                    "category": CATEGORY_MAP[img["label"]],
                    **result,
                }
                out.write(json.dumps(entry) + "\n")
                out.flush()

            except Exception as e:
                print(f"\nError processing {img['path']}: {e}")
                failed += 1

            time.sleep(args.delay)

    print(f"\nDone. Processed: {len(remaining) - failed}, Failed: {failed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
