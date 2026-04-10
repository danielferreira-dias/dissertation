"""
Stage 2: Reasoner — Generate structured clinical reasoning from observations.

Takes Stage 1 observations + ground-truth labels + images and generates
structured diagnostic reasoning using any vision-capable LLM via litellm.

Quick start (run after Stage 1 observer completes):
  python src/train/reasoner/main.py \\
      --model anthropic/claude-3-5-haiku-latest \\
      --observations data/reasoning/observations.jsonl

Test with 5 entries:
  python src/train/reasoner/main.py \\
      --model anthropic/claude-3-5-haiku-latest \\
      --limit 5

Other providers:
  python src/train/reasoner/main.py --model azure/gpt-4.1-mini
  python src/train/reasoner/main.py --model openai/gpt-4.1-mini
  python src/train/reasoner/main.py --model vertex_ai/gemini-2.5-pro
  python src/train/reasoner/main.py --model bedrock/anthropic.claude-3-5-haiku

Supported providers (set env vars):
  Vertex AI: gcloud auth application-default login (+ VERTEXAI_LOCATION=global)
  Azure:     AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
  AWS:       AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
  Anthropic: ANTHROPIC_API_KEY
  Google:    GEMINI_API_KEY
  OpenAI:    OPENAI_API_KEY

Input:   data/reasoning/observations.jsonl (from Stage 1)
Output:  data/reasoning/reasoning.jsonl
Flagged: data/reasoning/flagged_reasoner.jsonl
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Suppress litellm debug logging
litellm.suppress_debug_info = True

MAPPING_PATH = Path(__file__).resolve().parent.parent / "category_mapping.json"


def load_category_mapping() -> dict[str, str]:
    """Load the 335-class → 4-category mapping."""
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    mapping.pop("_meta", None)
    return mapping


REASONER_PROMPT = """You are a board-certified dermatologist generating structured clinical reasoning for a medical AI training dataset.

INPUTS:
- An image of a skin condition
- A visual observation (from a separate observer who did NOT know the diagnosis)
- The confirmed expert diagnosis: {diagnosis}
- Clinical category: {category}

TASK:
Using the image AND the observation, generate structured diagnostic reasoning that explains WHY the visual features support this diagnosis. Also provide differential diagnoses with contrastive reasoning (why those conditions are LESS likely).

RULES:
- The observation may miss features — look at the image yourself and add anything relevant.
- If the image clearly shows a DIFFERENT condition than {diagnosis}, set label_match to false.
- Adapt descriptions to the patient's skin tone.
- Be specific and clinical, not generic.
- Differentials should be conditions that look similar to this case, not random conditions.

VISUAL OBSERVATION FROM STAGE 1:
{observation}

Respond with ONLY a JSON object:
{{
  "morphology": "refined morphology connecting features to {diagnosis}",
  "color": "refined color description",
  "texture": "refined texture description",
  "border": "refined border description",
  "distribution": "anatomical location and pattern",
  "clinical_reasoning": "2-4 sentences explaining WHY these features support {diagnosis}",
  "differentials": [
    {{"condition": "most similar condition", "why_not": "why it's less likely than {diagnosis}"}},
    {{"condition": "second similar condition", "why_not": "why it's less likely"}}
  ],
  "confidence": "low | medium | high",
  "label_match": true or false
}}

No markdown, no code fences. Only the JSON object."""


def load_observations(path: Path) -> list[dict]:
    """Load Stage 1 observations."""
    entries = []
    with open(path) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


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


def format_observation(obs: dict) -> str:
    """Format Stage 1 observation dict into readable text."""
    parts = []
    for key in ["morphology", "color", "texture", "border", "distribution", "size_extent", "additional_features"]:
        val = obs.get(key, "")
        if val:
            parts.append(f"- {key.replace('_', ' ').title()}: {val}")
    return "\n".join(parts)


def generate_reasoning(
    image_path: str,
    label: str,
    category: str,
    observation: dict,
    model: str,
) -> tuple[dict | None, str | None]:
    """Send image + observation + label to any LLM via litellm.

    Returns (result, blocked_reason).
    """
    obs_text = format_observation(observation)
    prompt = REASONER_PROMPT.format(
        diagnosis=label.replace("_", " ").title(),
        category=category,
        observation=obs_text,
    )

    image_b64, mime = encode_image(image_path)

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}", "detail": "high"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.2,
        )
    except Exception as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in ["content_filter", "content management", "safety", "blocked"]):
            return None, f"content_filter | {str(e)[:200]}"
        raise

    choice = response.choices[0]

    if choice.finish_reason == "content_filter":
        return None, "content_filter | finish_reason=content_filter"

    text = choice.message.content
    if not text:
        return None, "empty_response"

    parsed = parse_response(text)
    if parsed is None:
        return None, "json_parse_error"
    return parsed, None


def _detect_source(image_path: str) -> str:
    """Detect which dataset an image came from based on path."""
    path_lower = image_path.lower()
    for source in ["fitzpatrick17k", "skincap", "dermnet_nz", "kaggle_dermnet", "pad_ufes", "scin"]:
        if source in path_lower:
            return source
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Reasoner — structured clinical reasoning via any LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model examples:
  --model azure/gpt-4.1-mini          Azure OpenAI
  --model anthropic/claude-3-5-haiku-latest  Anthropic direct
  --model bedrock/anthropic.claude-3-5-haiku AWS Bedrock
  --model gemini/gemini-2.5-flash      Google Gemini
  --model openai/gpt-4.1-mini         OpenAI direct
        """,
    )
    parser.add_argument("--observations", type=Path, default=Path("data/reasoning/observations.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reasoning/reasoning.jsonl"))
    parser.add_argument("--model", default="anthropic/claude-3-5-haiku-latest", help="litellm model string")
    parser.add_argument("--limit", type=int, default=None, help="Max entries to process")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between API calls")
    args = parser.parse_args()

    print(f"Model: {args.model}")

    category_mapping = load_category_mapping()

    observations = load_observations(args.observations)
    print(f"Loaded {len(observations)} observations")

    if args.limit:
        observations = observations[: args.limit]
        print(f"Limited to {len(observations)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_progress(args.output)
    remaining = [obs for obs in observations if obs["image_path"] not in done]
    print(f"Already processed: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    flagged_path = args.output.parent / "flagged_reasoner.jsonl"
    failed = 0
    flagged = 0
    with open(args.output, "a") as out, open(flagged_path, "a") as flagged_out:
        for obs in tqdm(remaining, desc="Generating reasoning"):
            image_path = obs["image_path"]
            label = obs["label"]
            category = category_mapping.get(label, "benign")
            observation = obs.get("observation", {})

            try:
                result, blocked_reason = generate_reasoning(
                    image_path, label, category, observation, args.model
                )

                if blocked_reason:
                    flagged += 1
                    flagged_entry = {
                        "image_path": image_path,
                        "label": label,
                        "category": category,
                        "reason": blocked_reason,
                    }
                    flagged_out.write(json.dumps(flagged_entry) + "\n")
                    flagged_out.flush()
                    print(f"\nFlagged: {image_path} — {blocked_reason}")
                    continue

                entry = {
                    "image_path": image_path,
                    "ground_truth": label,
                    "category": category,
                    "dataset_source": _detect_source(image_path),
                    "observation": observation,
                    "reasoning": result,
                }
                out.write(json.dumps(entry) + "\n")
                out.flush()

            except Exception as e:
                print(f"\nError: {image_path}: {e}")
                failed += 1

            time.sleep(args.delay)

    print(f"\nDone. Processed: {len(remaining) - failed - flagged}, Flagged: {flagged}, Failed: {failed}")
    print(f"Output: {args.output}")
    if flagged > 0:
        print(f"Flagged images: {flagged_path}")


if __name__ == "__main__":
    main()
