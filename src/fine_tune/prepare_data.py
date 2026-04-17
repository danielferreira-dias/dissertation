"""
Prepare training data in two formats for the ablation study.

Reads reasoning.jsonl and produces:
  - label_only/train.jsonl    — diagnosis + category only
  - full_reasoning/train.jsonl — complete structured reasoning output

Both formats use the Qwen VL chat template (image + text → assistant response).
A held-out validation split (10%) is created for early stopping.

Usage:
  python src/fine_tune/prepare_data.py
  python src/fine_tune/prepare_data.py --input data/reasoning/reasoning.jsonl --output-dir data/fine_tune
  python src/fine_tune/prepare_data.py --val-ratio 0.1 --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def format_observation_text(obs: dict) -> str:
    """Format observation dict into readable text."""
    parts = []
    for key in ["morphology", "color", "texture", "border", "distribution", "size_extent"]:
        val = obs.get(key, "")
        if val:
            parts.append(val)
    return " ".join(parts)


def make_label_only(entry: dict) -> dict:
    """Create a label-only training example (diagnosis + category)."""
    image_path = entry["image_path"]
    ground_truth = entry["ground_truth"]
    category = entry["category"]

    assistant_output = {
        "diagnosis": ground_truth.replace("_", " ").title(),
        "category": category,
    }

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Diagnose this skin condition."},
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_output),
            },
        ]
    }


def make_full_reasoning(entry: dict) -> dict:
    """Create a full reasoning training example (observations + reasoning + differentials)."""
    image_path = entry["image_path"]
    ground_truth = entry["ground_truth"]
    category = entry["category"]
    observation = entry.get("observation", {})
    reasoning = entry.get("reasoning", {})

    assistant_output = {
        "diagnosis": ground_truth.replace("_", " ").title(),
        "category": category,
        "observation": format_observation_text(observation),
        "morphology": reasoning.get("morphology", ""),
        "color": reasoning.get("color", ""),
        "texture": reasoning.get("texture", ""),
        "border": reasoning.get("border", ""),
        "distribution": reasoning.get("distribution", ""),
        "reasoning": reasoning.get("clinical_reasoning", ""),
        "differentials": reasoning.get("differentials", []),
        "confidence": reasoning.get("confidence", "medium"),
    }

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Diagnose this skin condition with structured reasoning."},
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_output),
            },
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning data in label-only and full-reasoning formats")
    parser.add_argument("--input", type=Path, default=Path("data/reasoning/reasoning.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/fine_tune"))
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    entries = [json.loads(line) for line in open(args.input)]
    print(f"Loaded {len(entries)} reasoning entries")

    random.seed(args.seed)
    random.shuffle(entries)

    split_idx = int(len(entries) * (1 - args.val_ratio))
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]
    print(f"Split: {len(train_entries)} train, {len(val_entries)} val")

    formats = {
        "label_only": make_label_only,
        "full_reasoning": make_full_reasoning,
    }

    for fmt_name, fmt_fn in formats.items():
        fmt_dir = args.output_dir / fmt_name
        fmt_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in [("train", train_entries), ("val", val_entries)]:
            out_path = fmt_dir / f"{split_name}.jsonl"
            with open(out_path, "w") as f:
                for entry in split_data:
                    f.write(json.dumps(fmt_fn(entry)) + "\n")
            print(f"  {fmt_name}/{split_name}.jsonl: {len(split_data)} examples")

    # Write metadata for reproducibility
    meta = {
        "source": str(args.input),
        "total_entries": len(entries),
        "train_size": len(train_entries),
        "val_size": len(val_entries),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "formats": list(formats.keys()),
    }
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
