"""
Convert raw Gemini descriptions (descriptions.jsonl) to VLM chat training format (train.jsonl).
"""

import argparse
import json
from pathlib import Path

LABEL_DISPLAY = {
    "melanoma": "Melanoma",
    "basal_cell_carcinoma": "Basal Cell Carcinoma",
    "seborrheic_keratosis": "Seborrheic Keratosis",
    "psoriasis": "Psoriasis",
    "eczema": "Eczema",
    "seborrheic_dermatitis": "Seborrheic Dermatitis",
}

CATEGORY_MAP = {
    "melanoma": "Malignant",
    "basal_cell_carcinoma": "Malignant",
    "seborrheic_keratosis": "Benign",
    "psoriasis": "Inflammatory",
    "eczema": "Inflammatory",
    "seborrheic_dermatitis": "Inflammatory",
}

USER_PROMPT = "Analyze this dermatological image. Provide the diagnosis, whether it is benign/malignant/inflammatory, your confidence level, the estimated Fitzpatrick skin type, and your clinical reasoning."


def convert_entry(entry: dict) -> dict:
    """Convert a raw description entry to chat training format."""
    label = entry["label"]
    diagnosis = LABEL_DISPLAY[label]
    category = CATEGORY_MAP[label]
    confidence = entry.get("confidence", "medium")
    fst = entry.get("fitzpatrick_skin_type", "unknown")
    reasoning = entry.get("reasoning", entry.get("clinical_reasoning", ""))

    assistant_response = (
        f"Diagnosis: {diagnosis}\n"
        f"Category: {category}\n"
        f"Confidence: {confidence.capitalize()}\n"
        f"Fitzpatrick skin type: {fst}\n"
        f"Reasoning: {reasoning}"
    )

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": entry["image_path"]},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": assistant_response,
            },
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Convert descriptions to VLM training format")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/reasoning/descriptions.jsonl"),
        help="Raw descriptions JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reasoning/train.jsonl"),
        help="Output training JSONL",
    )
    args = parser.parse_args()

    entries = []
    with open(args.input) as f:
        for line in f:
            entries.append(json.loads(line))

    matched = [e for e in entries if e.get("label_match", True)]
    skipped = len(entries) - len(matched)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for entry in matched:
            training_entry = convert_entry(entry)
            out.write(json.dumps(training_entry) + "\n")

    print(f"Converted {len(matched)} entries → {args.output}")
    if skipped:
        print(f"Skipped {skipped} mismatched labels")


if __name__ == "__main__":
    main()
