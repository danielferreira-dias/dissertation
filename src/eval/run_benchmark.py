"""
Run a model against a benchmark and save results using vLLM for fast inference.

Usage:
  python src/eval/run_benchmark.py --model gemma4-e4b --benchmark fitzpatrick17k
  python src/eval/run_benchmark.py --model qwen3.5-4b --benchmark mm_skin_vqa
  python src/eval/run_benchmark.py --all  # run all models on all benchmarks

Results saved to: final/results/<model>/<benchmark>/
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from config import (
    BENCHMARKS,
    CLASSIFICATION_PROMPT,
    CLASSIFICATION_SCHEMA,
    MODELS,
    RESULTS_DIR,
    TRIAD_PROMPT,
    TRIAD_SCHEMA,
    VQA_PROMPT_SUFFIX,
    VQA_SCHEMA,
)

BATCH_SIZE = 16
SAMPLE_LIMIT = 0  # 0 = no limit


def load_model(model_key: str):
    """Load model with vLLM and processor for chat template."""
    model_cfg = MODELS[model_key]
    hf_id = model_cfg["hf_id"]
    print(f"Loading {hf_id} with vLLM...")

    import os
    hf_token = os.environ.get("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True, token=hf_token)

    llm = LLM(
        model=hf_id,
        download_dir="/workspace/hf_cache",
        max_model_len=8192,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )
    print(f"Loaded {hf_id}")
    return llm, processor


def build_prompt(processor, image_path: str, prompt: str) -> tuple[str, Image.Image]:
    """Build chat prompt using the processor's template and load image."""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return chat_prompt, image


def run_batch(
    llm,
    prompts_and_images: list[tuple[str, Image.Image]],
    json_schema: dict | None = None,
    max_tokens: int = 2048,
) -> list[str]:
    """Run a batch of prompts through vLLM and return responses."""
    from vllm.sampling_params import StructuredOutputsParams

    structured = None
    if json_schema:
        structured = StructuredOutputsParams(json=json_schema)

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        structured_outputs=structured,
    )

    requests = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
        for prompt, image in prompts_and_images
    ]

    outputs = llm.generate(requests, sampling_params=sampling)
    return [out.outputs[0].text for out in outputs]


def parse_json_response(text: str) -> dict | None:
    """Try to parse JSON from model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def normalize_name(name: str) -> str:
    return str(name).lower().strip().replace(" ", "_").replace("-", "_").replace("/", "_")


# ── Benchmark runners ───────────────────────────────────────────────────────


def run_fitzpatrick(llm, processor, model_key: str):
    """Run Fitzpatrick17k classification benchmark."""
    bench = BENCHMARKS["fitzpatrick17k"]
    metadata = pd.read_csv(bench["metadata"])
    results_dir = RESULTS_DIR / model_key / "fitzpatrick17k"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "predictions.jsonl"
    done = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                entry = json.loads(line)
                done.add(entry.get("image_id"))

    remaining = metadata[~metadata["id"].astype(str).isin(done)]
    if SAMPLE_LIMIT > 0:
        remaining = remaining.head(SAMPLE_LIMIT)
    print(f"Fitzpatrick17k: {len(done)} done, {len(remaining)} remaining")

    # Build all prompts
    batch = []
    batch_meta = []
    processed = 0

    with open(results_file, "a") as out:
        for _, row in remaining.iterrows():
            disease = normalize_name(row["disease"])
            img_path = bench["path"] / disease / row["skincap_file_path"]

            if not img_path.exists():
                continue

            try:
                prompt, image = build_prompt(processor, str(img_path), CLASSIFICATION_PROMPT)
                batch.append((prompt, image))
                batch_meta.append(row)
            except Exception as e:
                print(f"\nSkipping {img_path}: {e}")
                continue

            if len(batch) >= BATCH_SIZE:
                responses = run_batch(llm, batch, json_schema=CLASSIFICATION_SCHEMA)
                for resp, meta_row in zip(responses, batch_meta):
                    parsed = parse_json_response(resp)
                    entry = {
                        "image_id": str(meta_row["id"]),
                        "ground_truth": meta_row["disease"],
                        "fitzpatrick_scale": meta_row.get("fitzpatrick_scale"),
                        "raw_response": resp,
                        "parsed": parsed,
                    }
                    out.write(json.dumps(entry) + "\n")
                out.flush()
                processed += len(batch)
                print(f"\r  Processed {processed}/{len(remaining)}", end="", flush=True)
                batch = []
                batch_meta = []

        # Process remaining
        if batch:
            responses = run_batch(llm, batch, json_schema=CLASSIFICATION_SCHEMA)
            for resp, meta_row in zip(responses, batch_meta):
                parsed = parse_json_response(resp)
                entry = {
                    "image_id": str(meta_row["id"]),
                    "ground_truth": meta_row["disease"],
                    "fitzpatrick_scale": meta_row.get("fitzpatrick_scale"),
                    "raw_response": resp,
                    "parsed": parsed,
                }
                out.write(json.dumps(entry) + "\n")
            out.flush()
            processed += len(batch)

    print(f"\nFitzpatrick17k: {processed} predictions saved to {results_file}")


def run_mm_skin_vqa(llm, processor, model_key: str):
    """Run MM-Skin VQA benchmark."""
    bench = BENCHMARKS["mm_skin_vqa"]
    vqa_df = pd.read_csv(bench["metadata"])
    results_dir = RESULTS_DIR / model_key / "mm_skin_vqa"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "predictions.jsonl"
    done = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                entry = json.loads(line)
                done.add(entry.get("index"))

    remaining_idx = [i for i in range(len(vqa_df)) if i not in done]
    if SAMPLE_LIMIT > 0:
        remaining_idx = remaining_idx[:SAMPLE_LIMIT]
    print(f"MM-Skin VQA: {len(done)} done, {len(remaining_idx)} remaining")

    batch = []
    batch_meta = []
    processed = 0

    with open(results_file, "a") as out:
        for idx in remaining_idx:
            row = vqa_df.iloc[idx]
            img_filename = str(row["image"]).replace("dataset/", "")
            img_path = bench["path"] / img_filename

            if not img_path.exists():
                continue

            try:
                vqa_prompt = row["question"] + VQA_PROMPT_SUFFIX
                prompt, image = build_prompt(processor, str(img_path), vqa_prompt)
                batch.append((prompt, image))
                batch_meta.append({"index": idx, "image": img_filename,
                                   "question": row["question"], "ground_truth": row["answer"]})
            except Exception as e:
                continue

            if len(batch) >= BATCH_SIZE:
                responses = run_batch(llm, batch, json_schema=VQA_SCHEMA)
                for resp, meta in zip(responses, batch_meta):
                    parsed = parse_json_response(resp)
                    prediction = parsed.get("answer", resp) if parsed else resp
                    entry = {**meta, "raw_response": resp, "prediction": prediction}
                    out.write(json.dumps(entry) + "\n")
                out.flush()
                processed += len(batch)
                print(f"\r  Processed {processed}/{len(remaining_idx)}", end="", flush=True)
                batch = []
                batch_meta = []

        if batch:
            responses = run_batch(llm, batch, json_schema=VQA_SCHEMA)
            for resp, meta in zip(responses, batch_meta):
                parsed = parse_json_response(resp)
                prediction = parsed.get("answer", resp) if parsed else resp
                entry = {**meta, "raw_response": resp, "prediction": prediction}
                out.write(json.dumps(entry) + "\n")
            out.flush()
            processed += len(batch)

    print(f"\nMM-Skin VQA: {processed} predictions saved to {results_file}")


def run_confusion_triads(llm, processor, model_key: str):
    """Run confusion triads evaluation."""
    bench = BENCHMARKS["confusion_triads"]
    results_dir = RESULTS_DIR / model_key / "confusion_triads"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "predictions.jsonl"
    done = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                entry = json.loads(line)
                done.add(entry.get("image_path"))

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    test_images = []
    for cls_dir in sorted(bench["path"].iterdir()):
        if not cls_dir.is_dir():
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in image_exts and str(f) not in done:
                test_images.append({"path": str(f), "label": cls_dir.name})

    if SAMPLE_LIMIT > 0:
        test_images = test_images[:SAMPLE_LIMIT]
    print(f"Confusion triads: {len(done)} done, {len(test_images)} remaining")

    batch = []
    batch_meta = []
    processed = 0

    with open(results_file, "a") as out:
        for img in test_images:
            try:
                prompt, image = build_prompt(processor, img["path"], TRIAD_PROMPT)
                batch.append((prompt, image))
                batch_meta.append(img)
            except Exception as e:
                continue

            if len(batch) >= BATCH_SIZE:
                responses = run_batch(llm, batch, json_schema=TRIAD_SCHEMA)
                for resp, meta in zip(responses, batch_meta):
                    parsed = parse_json_response(resp)
                    entry = {
                        "image_path": meta["path"],
                        "ground_truth": meta["label"],
                        "raw_response": resp,
                        "parsed": parsed,
                    }
                    out.write(json.dumps(entry) + "\n")
                out.flush()
                processed += len(batch)
                print(f"\r  Processed {processed}/{len(test_images)}", end="", flush=True)
                batch = []
                batch_meta = []

        if batch:
            responses = run_batch(llm, batch, json_schema=TRIAD_SCHEMA)
            for resp, meta in zip(responses, batch_meta):
                parsed = parse_json_response(resp)
                entry = {
                    "image_path": meta["path"],
                    "ground_truth": meta["label"],
                    "raw_response": resp,
                    "parsed": parsed,
                }
                out.write(json.dumps(entry) + "\n")
            out.flush()
            processed += len(batch)

    print(f"\nConfusion triads: {processed} predictions saved to {results_file}")


# ── Main ────────────────────────────────────────────────────────────────────

BENCHMARK_RUNNERS = {
    "fitzpatrick17k": run_fitzpatrick,
    "mm_skin_vqa": run_mm_skin_vqa,
    "confusion_triads": run_confusion_triads,
}


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation on dermatology benchmarks")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Model to evaluate")
    parser.add_argument("--benchmark", choices=list(BENCHMARKS.keys()), help="Benchmark to run")
    parser.add_argument("--all", action="store_true", help="Run all models on all benchmarks")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples per benchmark (0 = all)")
    args = parser.parse_args()

    global BATCH_SIZE, SAMPLE_LIMIT
    BATCH_SIZE = args.batch_size
    SAMPLE_LIMIT = args.limit

    if args.all:
        pairs = [(m, b) for m in MODELS for b in BENCHMARKS]
    elif args.model and args.benchmark:
        pairs = [(args.model, args.benchmark)]
    else:
        parser.error("Specify --model and --benchmark, or --all")

    current_model_key = None
    llm = None
    processor = None

    for model_key, bench_key in pairs:
        print(f"\n{'='*60}")
        print(f"Model: {model_key} | Benchmark: {bench_key}")
        print(f"{'='*60}")

        # Only reload model if it changed
        if model_key != current_model_key:
            if llm is not None:
                del llm
                import gc
                gc.collect()
            llm, processor = load_model(model_key)
            current_model_key = model_key

        t0 = time.time()
        runner = BENCHMARK_RUNNERS[bench_key]
        runner(llm, processor, model_key)
        elapsed = time.time() - t0
        print(f"  Time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
