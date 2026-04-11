"""
Convert Stage 2 reasoning output into fine-tuning conversation format.

Takes reasoning.jsonl and produces train.jsonl in the chat format
expected by Qwen 3.5 VL LoRA fine-tuning.

HYBRID LABEL AUTHORITY MODEL:
The ground-truth label is authoritative. Every reasoning entry is written as
supporting its label (the prompt forces this), so ALL entries are emitted to
the training file regardless of the reasoner's audit opinion. Entries where
the reasoner flagged a potential mismatch (label_match=false) are ALSO
written to a separate audit file for manual dataset-quality review.

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


def convert_entry(entry: dict) -> dict:
    """Convert a reasoning entry into a training conversation.

    All entries become training data — the hybrid model guarantees that the
    clinical_reasoning field supports the authoritative label regardless of
    the reasoner's audit opinion.
    """
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
    parser = argparse.ArgumentParser(description="Convert reasoning to training format")
    parser.add_argument("--input", type=Path, default=Path("data/reasoning/reasoning.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reasoning/train.jsonl"))
    parser.add_argument("--audit-output", type=Path, default=Path("data/reasoning/dataset_audit.jsonl"),
                        help="Where to write label_match=false audit entries")
    args = parser.parse_args()

    entries = [json.loads(line) for line in open(args.input)]
    print(f"Loaded {len(entries)} reasoning entries")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    audit_count = 0

    with open(args.output, "w") as out, open(args.audit_output, "w") as audit:
        for entry in entries:
            out.write(json.dumps(convert_entry(entry)) + "\n")
            converted += 1

            reasoning = entry.get("reasoning", {})
            if reasoning.get("label_match") is False:
                audit.write(json.dumps({
                    "image_path": entry["image_path"],
                    "ground_truth": entry["ground_truth"],
                    "dataset_source": entry.get("dataset_source", "unknown"),
                    "reasoner_opinion": reasoning.get("label_match_reason", ""),
                    "reasoner_confidence": reasoning.get("confidence"),
                    "top_differential": (reasoning.get("differentials") or [{}])[0],
                }) + "\n")
                audit_count += 1

    print(f"Converted to training: {converted}")
    print(f"Flagged for audit (label_match=false): {audit_count}")
    print(f"Training output: {args.output}")
    print(f"Audit output: {args.audit_output}")


if __name__ == "__main__":
    main()
