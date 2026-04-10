"""
Stage 1: Observer — Generate visual descriptions for dermatology images.

Describes images WITHOUT any diagnosis knowledge using any vision-capable LLM.

Quick start (tested config — Gemini 3 Flash on Vertex AI):
  export VERTEXAI_LOCATION=global
  python src/train/observer/main.py \\
      --model vertex_ai/gemini-3-flash-preview \\
      --data-dir final/train \\
      --delay 1

Test with 5 images:
  VERTEXAI_LOCATION=global python src/train/observer/main.py \\
      --model vertex_ai/gemini-3-flash-preview \\
      --data-dir final/train \\
      --limit 1 --classes melanoma psoriasis eczema basal_cell_carcinoma seborrheic_keratosis

Other providers:
  python src/train/observer/main.py --model anthropic/claude-3-5-haiku-latest
  python src/train/observer/main.py --model openai/gpt-4o-mini
  python src/train/observer/main.py --model azure/gpt-4o-mini
  python src/train/observer/main.py --model bedrock/amazon.nova-lite-v1:0

Supported providers (set env vars):
  Vertex AI: gcloud auth application-default login (+ VERTEXAI_LOCATION=global)
  Azure:     AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
  AWS:       AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
  Anthropic: ANTHROPIC_API_KEY
  Google:    GEMINI_API_KEY
  OpenAI:    OPENAI_API_KEY

Output: data/reasoning/observations.jsonl
Flagged: data/reasoning/flagged.jsonl
"""

import argparse
import base64
import json
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

litellm.suppress_debug_info = True

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


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and detect mime type."""
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


def observe_image(image_path: str, model: str) -> tuple[dict | None, str | None]:
    """Send image to any LLM via litellm and get pure visual description.

    Returns (result, blocked_reason).
    """
    image_b64, mime = encode_image(image_path)

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}", "detail": "high"}},
                        {"type": "text", "text": OBSERVER_PROMPT},
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.2,
        )
    except Exception as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in ["content_filter", "content management", "safety", "blocked", "recitation"]):
            return None, f"content_filter | {str(e)[:200]}"
        raise

    choice = response.choices[0]

    if choice.finish_reason in ("content_filter", "safety"):
        return None, f"content_filter | finish_reason={choice.finish_reason}"

    text = choice.message.content
    if not text:
        return None, "empty_response"

    parsed = parse_response(text)
    if parsed is None:
        return None, "json_parse_error"
    return parsed, None


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Observer — visual descriptions via any LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model examples:
  --model gemini/gemini-2.5-flash       Google Gemini
  --model gemini/gemini-3-flash         Google Gemini 3
  --model anthropic/claude-3-5-haiku-latest  Anthropic direct
  --model bedrock/anthropic.claude-3-5-haiku AWS Bedrock
  --model azure/gpt-4o-mini             Azure OpenAI
  --model openai/gpt-4o-mini            OpenAI direct
  --model bedrock/amazon.nova-lite-v1:0  AWS Nova Lite
        """,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("final/train"))
    parser.add_argument("--output", type=Path, default=Path("data/reasoning/observations.jsonl"))
    parser.add_argument("--model", default="vertex_ai/gemini-3-flash-preview", help="litellm model string")
    parser.add_argument("--limit", type=int, default=None, help="Max images per class")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--classes", nargs="+", default=None, help="Specific classes (default: all)")
    args = parser.parse_args()

    print(f"Model: {args.model}")

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
                result, blocked_reason = observe_image(img["path"], args.model)

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
