"""
Stage 2: Reasoner — Generate structured clinical reasoning from observations.

Takes Stage 1 observations + ground-truth labels + images and generates
structured diagnostic reasoning using any vision-capable LLM via litellm.

Default: Gemini 3.1 Pro Preview on Vertex AI with capped thinking budget.

Quick start:
  python3 src/train/reasoner/main.py --limit 5

Override model:
  python3 src/train/reasoner/main.py --model vertex_ai/gemini-3-flash-preview
  python3 src/train/reasoner/main.py --model anthropic/claude-sonnet-4-6
  python3 src/train/reasoner/main.py --model vertex_ai/gemini-3.1-pro-preview --thinking-budget 4096

Cap thinking tokens (Gemini only):
  --thinking-budget 2048   default; set 0 to disable thinking
  --max-tokens 4096        must be > thinking-budget + expected output

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

from dotenv import load_dotenv

load_dotenv()

# Vertex AI env setup (Gemini on dissertation-derm-vlm project)
os.environ.setdefault("VERTEXAI_PROJECT", os.environ.get("GCP_PROJECT", "dissertation-derm-vlm"))
os.environ.setdefault("VERTEXAI_LOCATION", "global")

# Clean up expired Bedrock bearer token so boto3/litellm use IAM creds
os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

import litellm
from tqdm import tqdm

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
- The CONFIRMED expert diagnosis (AUTHORITATIVE ground truth): {diagnosis}
- Clinical category: {category}

CRITICAL GROUND-TRUTH RULE:
The diagnosis {diagnosis} is AUTHORITATIVE and must be treated as correct. Your clinical_reasoning, morphology, color, texture, border, and distribution fields MUST be written as if the diagnosis is correct — they are training data for a student model that learns to output reasoning supporting the correct label. Do NOT caveat the reasoning. Do NOT write "this doesn't look like X". Even if features appear atypical, frame them as "atypical variant of {diagnosis}" or describe the features that WOULD be expected for {diagnosis} and note which are visible.

SEPARATELY, you may give your honest audit opinion ONLY in the label_match / label_match_reason fields (see below). These are for dataset quality review and are NOT used for training.

TASK:
1. Using the image AND the observation, generate structured diagnostic reasoning that explains WHY the features (visible and expected) support {diagnosis}.
2. Provide 3 differential diagnoses — conditions that look similar to {diagnosis} — with contrastive reasoning (why {diagnosis} is more likely than each).
3. Give an independent audit opinion on whether the image actually shows {diagnosis} (label_match + label_match_reason).

STYLE RULES:
- The Stage 1 observation may miss features — look at the image yourself and add anything relevant.
- Adapt descriptions to the patient's skin tone.
- Be specific and clinical, not generic. Vary your phrasing — avoid formulaic templates.
- DO NOT start clinical_reasoning with any of these phrases: "The image demonstrates...", "The image displays...", "The image shows...", "The presentation of...", "The clinical presentation of...", "The lesion presents as...". Start with something else — the anatomical feature, the diagnosis itself, the patient context, or a clinical observation.
- Differentials must be 3 DIFFERENT conditions genuinely similar to {diagnosis}, ranked by visual resemblance. Never duplicate {diagnosis} itself.

CONFIDENCE CALIBRATION (be honest — this reflects your confidence that the features described are visible in the image):
- "high": classic textbook features clearly visible. Use sparingly.
- "medium": features consistent with the diagnosis but image quality or atypical presentation limits certainty.
- "low": poor image quality or features are difficult to discern.

VISUAL OBSERVATION FROM STAGE 1:
{observation}

Respond with ONLY a JSON object:
{{
  "morphology": "refined morphology (written as supporting {diagnosis})",
  "color": "refined color description",
  "texture": "refined texture description",
  "border": "refined border description",
  "distribution": "anatomical location and pattern",
  "clinical_reasoning": "3 to 5 sentences explaining WHY the features (visible and expected) support {diagnosis}. Include distinguishing features, anatomical/epidemiological context, and at least one nuance specific to this image. Do NOT question the label here.",
  "differentials": [
    {{"condition": "closest look-alike condition", "why_not": "specific visual/clinical feature that distinguishes {diagnosis} from this"}},
    {{"condition": "second closest look-alike", "why_not": "specific distinguishing feature"}},
    {{"condition": "third closest look-alike", "why_not": "specific distinguishing feature"}}
  ],
  "confidence": "low | medium | high",
  "label_match": true or false,
  "label_match_reason": "Brief honest opinion ONLY if label_match=false. Explain what the image actually appears to show and why the provided diagnosis {diagnosis} seems inconsistent. Leave empty string if label_match=true. This field is for dataset audit only."
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


def load_flagged_as_observations(path: Path) -> list[dict]:
    """Load Stage 1 flagged entries and shape them like observations with empty observation dict.

    These are images where the Gemini Flash observer failed (usually json_parse_error).
    Gemini 3.1 Pro will observe AND reason in one pass when it sees an empty observation.
    """
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            try:
                flagged = json.loads(line)
                entries.append({
                    "image_path": flagged["image_path"],
                    "label": flagged["label"],
                    "observation": {},
                    "_from_flagged": True,
                })
            except (json.JSONDecodeError, KeyError):
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
    """Parse JSON response, handling markdown fences and list-wrapped objects.

    Some Gemini calls return [{...}] instead of {...} — unwrap single-element
    lists whose sole element is a dict.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        return obj[0]
    if isinstance(obj, dict):
        return obj
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
    if not parts:
        return "(No Stage 1 observation available — the Stage 1 observer failed for this image. Observe the image yourself and generate all fields from direct inspection.)"
    return "\n".join(parts)


RESPONSE_FORMAT = {"type": "json_object"}

MAX_RETRIES = 3


def _try_single_model(
    model: str,
    prompt: str,
    image_b64: str,
    mime: str,
    max_tokens: int,
    thinking_budget: int,
    temperature: float,
) -> tuple[dict | None, str | None]:
    """Try one model with retries. Returns (parsed_dict, error_reason).

    error_reason categories:
      - "content_filter | ..."     -> fatal for this model, do not retry with same model
      - "model_unavailable | ..."  -> model is unreachable; caller should try fallback
      - "max_retries_exceeded"     -> transient failures exhausted retries; caller may fall back
      - "json_parse_error" / "empty_response" -> parse issues after retries
    """
    kwargs = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}", "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": RESPONSE_FORMAT,
    }
    # Gemini thinking-budget control (Vertex AI Gemini 2.5+ / 3.x)
    if "gemini" in model.lower() and thinking_budget >= 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(**kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ["content_filter", "content management", "safety", "blocked"]):
                return None, f"content_filter | {str(e)[:200]}"
            if any(kw in error_msg for kw in ["404", "not found", "model not available", "deprecated"]):
                return None, f"model_unavailable | {str(e)[:200]}"
            if any(kw in error_msg for kw in ["rate limit", "ratelimit", "too many requests", "429", "503", "quota", "overloaded"]) and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return None, f"transient_error | {str(e)[:200]}"

        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            return None, "content_filter | finish_reason=content_filter"

        text = choice.message.content
        if not text:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return None, "empty_response"

        parsed = parse_response(text)
        if parsed is None:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return None, "json_parse_error"
        return parsed, None

    return None, "max_retries_exceeded"


# Errors that make a model worth falling back from (as opposed to surfacing as a permanent failure)
_FALLBACKABLE_REASONS = {"model_unavailable", "transient_error", "empty_response", "json_parse_error", "max_retries_exceeded"}


def _is_fallbackable(reason: str | None) -> bool:
    if not reason:
        return False
    head = reason.split(" | ", 1)[0]
    return head in _FALLBACKABLE_REASONS


def generate_reasoning(
    image_path: str,
    label: str,
    category: str,
    observation: dict,
    models: list[str],
    max_tokens: int,
    thinking_budget: int,
    temperature: float = 0.4,
) -> tuple[dict | None, str | None, str | None]:
    """Try models in priority order until one succeeds.

    Returns (result, blocked_reason, model_used). `model_used` is the string of
    whichever model successfully produced the parsed output; None on failure.
    Content-filter blocks are fatal (same content will likely be blocked by
    fallback too); all other failure categories trigger fallback to the next model.
    """
    obs_text = format_observation(observation)
    prompt = REASONER_PROMPT.format(
        diagnosis=label.replace("_", " ").title(),
        category=category,
        observation=obs_text,
    )
    image_b64, mime = encode_image(image_path)

    last_reason = None
    tried = []
    for model in models:
        tried.append(model)
        result, reason = _try_single_model(
            model, prompt, image_b64, mime, max_tokens, thinking_budget, temperature,
        )
        if result is not None:
            return result, None, model
        last_reason = reason
        # content_filter is not recoverable by retrying on a different model
        if reason and reason.startswith("content_filter"):
            return None, reason, None
        if not _is_fallbackable(reason):
            return None, reason, None
        # else fall through to next model

    return None, f"{last_reason} (tried: {','.join(tried)})", None

    return None, "max_retries_exceeded"


import re as _re

_SCIN_RE = _re.compile(r"^-?\d{10,}\.(png|jpg|jpeg)$", _re.IGNORECASE)
_FITZ_RE = _re.compile(r"^\d{1,6}\.(png|jpg|jpeg)$", _re.IGNORECASE)
_PAD_UFES_RE = _re.compile(r"^PAT_\d+", _re.IGNORECASE)
_DERMNET_NZ_RE = _re.compile(r"^[a-z][a-z0-9\-]*-\d+\.(png|jpg|jpeg)$", _re.IGNORECASE)
_KAGGLE_DERMNET_RE = _re.compile(r"^\d{1,3}[A-Za-z]")


def _detect_source(image_path: str) -> str:
    """Detect which dataset an image came from based on filename heuristics.

    The data-collection pipeline flattened dataset provenance — all images live
    under final/train/<class>/ regardless of source. We recover the source
    from filename patterns that are distinctive per dataset.
    """
    path_lower = image_path.lower()
    # Path-level tokens first (future-proofing)
    for source in ["fitzpatrick17k", "skincap", "dermnet_nz", "kaggle_dermnet", "pad_ufes", "scin"]:
        if source in path_lower:
            return source

    name = Path(image_path).name
    if _PAD_UFES_RE.match(name):
        return "pad_ufes"
    if _SCIN_RE.match(name):
        return "scin"
    if _FITZ_RE.match(name):
        return "fitzpatrick17k"
    if _DERMNET_NZ_RE.match(name):
        return "dermnet_nz"
    if _KAGGLE_DERMNET_RE.match(name):
        return "kaggle_dermnet"
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
    parser.add_argument("--model", default="vertex_ai/gemini-3.1-pro-preview", help="Primary litellm model string")
    parser.add_argument("--fallback-model", default="azure/grok-4-20-reasoning",
                        help="Fallback model if the primary is unreachable / fails / is removed. Pass empty string to disable.")
    parser.add_argument("--limit", type=int, default=None, help="Max entries to process")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between API calls")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--retry-flagged", action="store_true", help="Retry previously flagged images")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens (includes thinking tokens for Gemini)")
    parser.add_argument("--thinking-budget", type=int, default=2048, help="Gemini thinking budget cap (set to 0 to disable)")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Shuffle observations with this seed before applying --limit (for diverse pilots)")
    args = parser.parse_args()

    models_chain = [args.model] + ([args.fallback_model] if args.fallback_model else [])
    print(f"Models (primary -> fallback): {' -> '.join(models_chain)}")
    print(f"Workers: {args.workers} | max_tokens: {args.max_tokens} | thinking_budget: {args.thinking_budget} | temp: {args.temperature}")

    category_mapping = load_category_mapping()

    observations = load_observations(args.observations)
    print(f"Loaded {len(observations)} observations from {args.observations.name}")

    # Also absorb Stage 1 flagged entries (failed observer). Reasoner will observe + reason in one shot.
    observer_flagged_path = args.observations.parent / "flagged.jsonl"
    flagged_obs = load_flagged_as_observations(observer_flagged_path)
    if flagged_obs:
        print(f"Loaded {len(flagged_obs)} flagged Stage 1 entries (empty observations, reasoner will observe too)")
        observations.extend(flagged_obs)

    if args.shuffle_seed is not None:
        import random
        random.Random(args.shuffle_seed).shuffle(observations)
        print(f"Shuffled with seed={args.shuffle_seed}")

    if args.limit:
        observations = observations[: args.limit]
        print(f"Limited to {len(observations)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    flagged_path = args.output.parent / "flagged_reasoner.jsonl"

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
        remaining = [obs for obs in observations if obs["image_path"] not in done or obs["image_path"] in flagged_paths]
    else:
        remaining = [obs for obs in observations if obs["image_path"] not in done]

    print(f"Already processed: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    failed = 0
    flagged_count = 0
    processed = 0
    lock = __import__("threading").Lock()

    def process_observation(obs: dict) -> None:
        nonlocal failed, flagged_count, processed
        image_path = obs["image_path"]
        label = obs["label"]
        category = category_mapping.get(label, "benign")
        observation = obs.get("observation", {})

        try:
            result, blocked_reason, model_used = generate_reasoning(
                image_path, label, category, observation, models_chain,
                max_tokens=args.max_tokens, thinking_budget=args.thinking_budget,
                temperature=args.temperature,
            )

            with lock:
                if blocked_reason:
                    flagged_count += 1
                    flagged_entry = {
                        "image_path": image_path,
                        "label": label,
                        "category": category,
                        "reason": blocked_reason,
                    }
                    flagged_out.write(json.dumps(flagged_entry) + "\n")
                    flagged_out.flush()
                else:
                    entry = {
                        "image_path": image_path,
                        "ground_truth": label,
                        "category": category,
                        "dataset_source": _detect_source(image_path),
                        "model_used": model_used,
                        "observation": observation,
                        "reasoning": result,
                    }
                    out.write(json.dumps(entry) + "\n")
                    out.flush()
                    processed += 1

        except Exception as e:
            with lock:
                failed += 1
                tqdm.write(f"Error: {image_path}: {e}")

        time.sleep(args.delay)

    from concurrent.futures import ThreadPoolExecutor

    with open(args.output, "a") as out, open(flagged_path, "a") as flagged_out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(tqdm(
                pool.map(process_observation, remaining),
                total=len(remaining),
                desc="Generating reasoning",
            ))

    print(f"\nDone. Processed: {processed}, Flagged: {flagged_count}, Failed: {failed}")
    print(f"Output: {args.output}")
    if flagged_count > 0:
        print(f"Flagged images: {flagged_path}")


if __name__ == "__main__":
    main()
