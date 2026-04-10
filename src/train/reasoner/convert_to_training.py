"""
Convert Stage 2 reasoning output into fine-tuning conversation format.

Takes reasoning.jsonl and produces train.jsonl in the chat format
expected by Qwen 3.5 VL LoRA fine-tuning.

Usage:
  python src/train/reasoner/convert_to_training.py
  python src/train/reasoner/convert_to_training.py --input data/reasoning/reasoning.jsonl --output data/reasoning/train.jsonl
"""

import argparse
import json
from pathlib import Path


def format_observation_text(obs: dict) -> str:
    """Format observation dict into readable text for the assistant response."""
    parts = []
    for key in ["morphology", "color", "texture", "border", "distribution", "size_extent"]:
        val = obs.get(key, "")
        if val:
            parts.append(val)
    return " ".join(parts)


def convert_entry(entry: dict) -> dict | None:
    """Convert a reasoning entry into a training conversation."""
    image_path = entry["image_path"]
    ground_truth = entry["ground_truth"]
    category = entry["category"]
    observation = entry.get("observation", {})
    reasoning = entry.get("reasoning", {})

    # Skip bad labels flagged by reasoner
    if not reasoning.get("label_match", True):
        return None

    # Build assistant response JSON
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

    conversation = {
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

    return conversation


def main():
    parser = argparse.ArgumentParser(description="Convert reasoning to training format")
    parser.add_argument("--input", type=Path, default=Path("data/reasoning/reasoning.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reasoning/train.jsonl"))
    args = parser.parse_args()

    entries = [json.loads(line) for line in open(args.input)]
    print(f"Loaded {len(entries)} reasoning entries")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0

    with open(args.output, "w") as out:
        for entry in entries:
            conversation = convert_entry(entry)
            if conversation is None:
                skipped += 1
                continue
            out.write(json.dumps(conversation) + "\n")
            converted += 1

    print(f"Converted: {converted}, Skipped (label_match=false): {skipped}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
