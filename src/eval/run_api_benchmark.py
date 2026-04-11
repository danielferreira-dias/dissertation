"""
Run API-based vision models against Fitzpatrick17k and Confusion Triads benchmarks.

Plug in any model via litellm. Results are compatible with score_results.py.

Quick start:
  python3 src/eval/run_api_benchmark.py --model azure/grok-4-20-reasoning --benchmark confusion_triads --limit 5 --workers 2 --delay 1
  python3 src/eval/run_api_benchmark.py --model azure/grok-4-20-reasoning --benchmark all --workers 4 --delay 1
  python3 src/eval/run_api_benchmark.py --model xai/grok-4.20-beta --benchmark all --workers 8
  python3 src/eval/run_api_benchmark.py --model openai/gpt-5.4 --benchmark confusion_triads --limit 10
  python3 src/eval/run_api_benchmark.py --model anthropic/claude-3-5-haiku-latest --benchmark fitzpatrick17k

Score results:
  python3 src/eval/score_results.py

Supported providers (set env vars):
  OpenAI:    OPENAI_API_KEY
  Anthropic: ANTHROPIC_API_KEY
  xAI:       XAI_API_KEY
  Azure:     AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
  Azure AI:  AZURE_AI_FOUNDRY_API_BASE, AZURE_AI_FOUNDRY_API_KEY (for Grok, Llama, etc.)
  AWS:       AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
  Google:    GEMINI_API_KEY
  Vertex AI: gcloud auth application-default login

Script is resumable — re-run to continue from where it left off.
"""

import argparse
import base64
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
litellm.suppress_debug_info = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CLASSIFICATION_PROMPT = """Diagnose this skin condition. Respond with a JSON object:
{
  "diagnosis": "most likely condition",
  "top_6": ["1st most likely", "2nd most likely", "3rd", "4th", "5th", "6th"],
  "confidence": "low | medium | high",
  "reasoning": "brief explanation of visual features"
}
Each entry in top_6 must be a different skin condition name."""

TRIAD_PROMPT = """Which of these 6 conditions does this image most likely show?
1. seborrheic_dermatitis
2. psoriasis
3. eczema
4. seborrheic_keratosis
5. basal_cell_carcinoma
6. melanoma

Respond with a JSON object:
{
  "diagnosis": "one of the 6 conditions above",
  "confidence": "low | medium | high",
  "reasoning": "brief explanation of visual features"
}"""

RESPONSE_FORMAT = {"type": "json_object"}
MAX_RETRIES = 3


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and detect mime type."""
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


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


def call_model(image_path: str, prompt: str, model: str) -> tuple[dict | None, str]:
    """Call any vision model via litellm with retry logic.

    Returns (parsed_json, raw_response_text).
    """
    image_b64, mime = encode_image(image_path)

    for attempt in range(MAX_RETRIES):
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
                response_format=RESPONSE_FORMAT,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ["content_filter", "safety", "blocked"]):
                return None, f"BLOCKED: {str(e)[:200]}"
            if any(kw in error_msg for kw in ["rate limit", "429", "503", "quota", "overloaded"]) and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise

        text = response.choices[0].message.content or ""
        parsed = parse_response(text)
        if parsed is not None:
            return parsed, text
        if attempt < MAX_RETRIES - 1:
            time.sleep(1)
            continue

    return None, text


def normalize_name(name: str) -> str:
    return str(name).lower().strip().replace(" ", "_").replace("-", "_").replace("/", "_")


def load_progress(results_file: Path) -> set[str]:
    """Load already-processed IDs from existing results."""
    done = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done.add(entry.get("image_id") or entry.get("image_path", ""))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# ── Benchmark runners ───────────────────────────────────────────────────────


def run_fitzpatrick(model: str, output_dir: Path, workers: int, delay: float, limit: int):
    """Run Fitzpatrick17k benchmark via API."""
    bench_path = PROJECT_ROOT / "final" / "benchmarks" / "fitzpatrick17k_1000"
    metadata = pd.read_csv(bench_path / "benchmark_metadata.csv")
    results_dir = output_dir / "fitzpatrick17k"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "predictions.jsonl"

    done = load_progress(results_file)
    remaining = metadata[~metadata["id"].astype(str).isin(done)]
    if limit:
        remaining = remaining.head(limit)
    print(f"Fitzpatrick17k: {len(done)} done, {len(remaining)} remaining")

    if remaining.empty:
        print("Nothing to do.")
        return

    processed = 0
    failed = 0
    lock = threading.Lock()

    def process_row(row):
        nonlocal processed, failed
        disease = normalize_name(row["disease"])
        img_path = bench_path / disease / row["skincap_file_path"]
        if not img_path.exists():
            return

        try:
            parsed, raw = call_model(str(img_path), CLASSIFICATION_PROMPT, model)
            entry = {
                "image_id": str(row["id"]),
                "ground_truth": row["disease"],
                "fitzpatrick_scale": row.get("fitzpatrick_scale"),
                "raw_response": raw,
                "parsed": parsed,
            }
            with lock:
                out.write(json.dumps(entry) + "\n")
                out.flush()
                processed += 1
        except Exception as e:
            with lock:
                failed += 1
                tqdm.write(f"Error: {img_path}: {e}")

        time.sleep(delay)

    rows = [row for _, row in remaining.iterrows()]
    with open(results_file, "a") as out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(tqdm(pool.map(process_row, rows), total=len(rows), desc="Fitzpatrick17k"))

    print(f"Fitzpatrick17k: {processed} processed, {failed} failed")


def run_confusion_triads(model: str, output_dir: Path, workers: int, delay: float, limit: int):
    """Run confusion triads benchmark via API."""
    test_path = PROJECT_ROOT / "final" / "test"
    results_dir = output_dir / "confusion_triads"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "predictions.jsonl"

    done = load_progress(results_file)

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    test_images = []
    for cls_dir in sorted(test_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in image_exts and str(f) not in done:
                test_images.append({"path": str(f), "label": cls_dir.name})

    if limit:
        test_images = test_images[:limit]
    print(f"Confusion triads: {len(done)} done, {len(test_images)} remaining")

    if not test_images:
        print("Nothing to do.")
        return

    processed = 0
    failed = 0
    lock = threading.Lock()

    def process_image(img):
        nonlocal processed, failed
        try:
            parsed, raw = call_model(img["path"], TRIAD_PROMPT, model)
            entry = {
                "image_path": img["path"],
                "ground_truth": img["label"],
                "raw_response": raw,
                "parsed": parsed,
            }
            with lock:
                out.write(json.dumps(entry) + "\n")
                out.flush()
                processed += 1
        except Exception as e:
            with lock:
                failed += 1
                tqdm.write(f"Error: {img['path']}: {e}")

        time.sleep(delay)

    with open(results_file, "a") as out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(tqdm(pool.map(process_image, test_images), total=len(test_images), desc="Confusion triads"))

    print(f"Confusion triads: {processed} processed, {failed} failed")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run API vision models against dermatology benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model examples:
  --model xai/grok-4.20-beta              xAI Grok
  --model openai/gpt-5.4                  OpenAI GPT-5.4
  --model openai/gpt-5.4-mini             OpenAI GPT-5.4 Mini
  --model anthropic/claude-3-5-haiku-latest  Anthropic Haiku
  --model azure/your-deployment            Azure OpenAI
  --model gemini/gemini-2.5-pro            Google Gemini
        """,
    )
    parser.add_argument("--model", required=True, help="litellm model string")
    parser.add_argument("--benchmark", choices=["fitzpatrick17k", "confusion_triads", "all"], default="all")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--limit", type=int, default=0, help="Max images per benchmark (0 = all)")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between API calls")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: final/results/<model-name>/)")
    args = parser.parse_args()

    # Derive model name for output directory
    model_name = args.model.split("/")[-1]
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "final" / "results" / model_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    limit = args.limit if args.limit > 0 else None

    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print()

    if args.benchmark in ("fitzpatrick17k", "all"):
        run_fitzpatrick(args.model, args.output_dir, args.workers, args.delay, limit)
        print()

    if args.benchmark in ("confusion_triads", "all"):
        run_confusion_triads(args.model, args.output_dir, args.workers, args.delay, limit)
        print()

    print("Done. Score with: python3 src/eval/score_results.py")


if __name__ == "__main__":
    main()
