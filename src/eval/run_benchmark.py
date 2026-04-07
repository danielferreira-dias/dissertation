"""
Run a model against a benchmark and save results.

Usage:
  python src/eval/run_benchmark.py --model medgemma-4b --benchmark fitzpatrick17k
  python src/eval/run_benchmark.py --model qwen3.5-4b --benchmark mm_skin_vqa
  python src/eval/run_benchmark.py --model gemma4-e4b --benchmark confusion_triads
  python src/eval/run_benchmark.py --all  # run all models on all benchmarks

Results saved to: final/results/<model>/<benchmark>/
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from config import BENCHMARKS, CLASSIFICATION_PROMPT, MODELS, RESULTS_DIR, TRIAD_PROMPT


def load_model(model_key: str):
    """Load model and processor from HuggingFace."""
    model_cfg = MODELS[model_key]
    hf_id = model_cfg["hf_id"]
    print(f"Loading {hf_id}...")

    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded {hf_id} on {model.device}")
    return model, processor


def run_inference(model, processor, image_path: str, prompt: str) -> str:
    """Run a single inference and return the model's text response."""
    image = Image.open(image_path)
    image.verify()  # Check image is valid
    image = Image.open(image_path).convert("RGB")  # Reopen after verify

    # Build messages in chat format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process inputs — handle different model architectures
    try:
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
    except Exception:
        # Fallback for models that need different image processing
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt",
        )

    # Move to device, only tensor values
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Filter out kwargs the model doesn't accept
    input_ids = inputs.pop("input_ids")
    attention_mask = inputs.pop("attention_mask", None)

    generate_kwargs = {"input_ids": input_ids, "max_new_tokens": 512, "do_sample": False}
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    # Pass remaining inputs (pixel_values, image_grid_thw, etc.) only if model accepts them
    import inspect
    model_forward_params = set(inspect.signature(model.forward).parameters.keys())
    for k, v in inputs.items():
        if k in model_forward_params:
            generate_kwargs[k] = v

    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)

    # Decode only the generated tokens
    generated = output_ids[0][input_ids.shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)
    return response


def parse_json_response(text: str) -> dict | None:
    """Try to parse JSON from model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        import re
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ── Benchmark runners ───────────────────────────────────────────────────────


def run_fitzpatrick(model, processor, model_key: str):
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
    print(f"Fitzpatrick17k: {len(done)} done, {len(remaining)} remaining")

    def normalize(name):
        return str(name).lower().strip().replace(" ", "_").replace("-", "_").replace("/", "_")

    with open(results_file, "a") as out:
        for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Fitzpatrick17k"):
            disease = normalize(row["disease"])
            img_path = bench["path"] / disease / row["skincap_file_path"]

            if not img_path.exists():
                continue

            try:
                response = run_inference(model, processor, str(img_path), CLASSIFICATION_PROMPT)
                parsed = parse_json_response(response)

                entry = {
                    "image_id": str(row["id"]),
                    "ground_truth": row["disease"],
                    "fitzpatrick_scale": row.get("fitzpatrick_scale"),
                    "raw_response": response,
                    "parsed": parsed,
                }
                out.write(json.dumps(entry) + "\n")
                out.flush()
            except Exception as e:
                print(f"\nError: {e}")

    print(f"Results saved to {results_file}")


def run_mm_skin_vqa(model, processor, model_key: str):
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
    print(f"MM-Skin VQA: {len(done)} done, {len(remaining_idx)} remaining")

    with open(results_file, "a") as out:
        for idx in tqdm(remaining_idx, desc="MM-Skin VQA"):
            row = vqa_df.iloc[idx]
            img_filename = str(row["image"]).replace("dataset/", "")
            img_path = bench["path"] / img_filename

            if not img_path.exists():
                continue

            question = row["question"]
            ground_truth = row["answer"]

            try:
                response = run_inference(model, processor, str(img_path), question)

                entry = {
                    "index": idx,
                    "image": img_filename,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": response,
                }
                out.write(json.dumps(entry) + "\n")
                out.flush()
            except Exception as e:
                print(f"\nError: {e}")

    print(f"Results saved to {results_file}")


def run_confusion_triads(model, processor, model_key: str):
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

    # Collect all test images
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    test_images = []
    for cls_dir in sorted(bench["path"].iterdir()):
        if not cls_dir.is_dir():
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in image_exts and str(f) not in done:
                test_images.append({"path": str(f), "label": cls_dir.name})

    print(f"Confusion triads: {len(done)} done, {len(test_images)} remaining")

    with open(results_file, "a") as out:
        for img in tqdm(test_images, desc="Confusion triads"):
            try:
                response = run_inference(model, processor, img["path"], TRIAD_PROMPT)
                parsed = parse_json_response(response)

                entry = {
                    "image_path": img["path"],
                    "ground_truth": img["label"],
                    "raw_response": response,
                    "parsed": parsed,
                }
                out.write(json.dumps(entry) + "\n")
                out.flush()
            except Exception as e:
                print(f"\nError: {e}")

    print(f"Results saved to {results_file}")


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
    args = parser.parse_args()

    if args.all:
        pairs = [(m, b) for m in MODELS for b in BENCHMARKS]
    elif args.model and args.benchmark:
        pairs = [(args.model, args.benchmark)]
    else:
        parser.error("Specify --model and --benchmark, or --all")

    for model_key, bench_key in pairs:
        print(f"\n{'='*60}")
        print(f"Model: {model_key} | Benchmark: {bench_key}")
        print(f"{'='*60}")

        model, processor = load_model(model_key)
        runner = BENCHMARK_RUNNERS[bench_key]
        runner(model, processor, model_key)

        # Free GPU memory between models
        del model
        del processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
