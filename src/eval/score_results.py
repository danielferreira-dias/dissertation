"""
Score benchmark results and generate comparison tables.

Usage:
  python src/eval/score_results.py                    # score all available results
  python src/eval/score_results.py --model medgemma-4b # score one model
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from config import MODELS, RESULTS_DIR


def normalize_diagnosis(text: str) -> str:
    """Normalize a diagnosis string for comparison."""
    return (
        text.lower()
        .strip()
        .replace("-", " ")
        .replace("_", " ")
        .replace("'s", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )


def score_fitzpatrick(results_file: Path) -> dict:
    """Score Fitzpatrick17k classification results."""
    entries = []
    with open(results_file) as f:
        for line in f:
            entries.append(json.loads(line))

    if not entries:
        return {}

    total = len(entries)
    top1_correct = 0
    top6_correct = 0
    fst_scores = defaultdict(lambda: {"total": 0, "top1": 0, "top6": 0})

    for e in entries:
        gt = normalize_diagnosis(e["ground_truth"])
        parsed = e.get("parsed")
        fst = e.get("fitzpatrick_scale")

        if parsed:
            pred = normalize_diagnosis(parsed.get("diagnosis", ""))
            top_n_raw = parsed.get("top_n") or parsed.get("top_6") or []
            top6 = [normalize_diagnosis(d) for d in top_n_raw]

            if gt in pred or pred in gt:
                top1_correct += 1
            if any(gt in d or d in gt for d in [pred] + top6):
                top6_correct += 1

        if fst and fst > 0:
            fst_key = int(fst)
            fst_scores[fst_key]["total"] += 1
            if parsed:
                pred = normalize_diagnosis(parsed.get("diagnosis", ""))
                if gt in pred or pred in gt:
                    fst_scores[fst_key]["top1"] += 1

    results = {
        "total": total,
        "top1_accuracy": top1_correct / total * 100 if total else 0,
        "top6_accuracy": top6_correct / total * 100 if total else 0,
        "top1_correct": top1_correct,
        "top6_correct": top6_correct,
        "fairness_by_fst": {},
    }

    for fst, scores in sorted(fst_scores.items()):
        results["fairness_by_fst"][f"FST_{fst}"] = {
            "total": scores["total"],
            "top1_accuracy": scores["top1"] / scores["total"] * 100 if scores["total"] else 0,
        }

    return results


def score_confusion_triads(results_file: Path) -> dict:
    """Score confusion triad classification results."""
    entries = []
    with open(results_file) as f:
        for line in f:
            entries.append(json.loads(line))

    if not entries:
        return {}

    total = len(entries)
    correct = 0
    confusion = defaultdict(Counter)
    class_scores = defaultdict(lambda: {"total": 0, "correct": 0})

    label_map = {
        "seborrheic dermatitis": "seborrheic_dermatitis",
        "psoriasis": "psoriasis",
        "eczema": "eczema",
        "seborrheic keratosis": "seborrheic_keratosis",
        "basal cell carcinoma": "basal_cell_carcinoma",
        "melanoma": "melanoma",
    }

    for e in entries:
        gt = e["ground_truth"]
        parsed = e.get("parsed")

        class_scores[gt]["total"] += 1

        if parsed:
            # Guided JSON uses enum values directly (e.g. "seborrheic_dermatitis")
            pred_raw = parsed.get("diagnosis", "")
            pred = label_map.get(normalize_diagnosis(pred_raw), pred_raw)

            if pred == gt or gt in pred or pred in gt:
                correct += 1
                class_scores[gt]["correct"] += 1

            confusion[gt][pred] += 1

    results = {
        "total": total,
        "accuracy": correct / total * 100 if total else 0,
        "correct": correct,
        "per_class": {},
        "confusion_matrix": {},
    }

    for cls, scores in sorted(class_scores.items()):
        results["per_class"][cls] = {
            "total": scores["total"],
            "accuracy": scores["correct"] / scores["total"] * 100 if scores["total"] else 0,
        }

    for gt, preds in confusion.items():
        results["confusion_matrix"][gt] = dict(preds.most_common(6))

    return results


def score_vqa(results_file: Path) -> dict:
    """Score MM-Skin VQA results (basic metrics)."""
    entries = []
    with open(results_file) as f:
        for line in f:
            entries.append(json.loads(line))

    if not entries:
        return {}

    total = len(entries)

    # Exact match (strict)
    exact = sum(
        1 for e in entries
        if normalize_diagnosis(e.get("prediction", "")) == normalize_diagnosis(e["ground_truth"])
    )

    # Containment match (lenient)
    contain = sum(
        1 for e in entries
        if normalize_diagnosis(e["ground_truth"]) in normalize_diagnosis(e.get("prediction", ""))
    )

    # BERTScore (semantic similarity)
    bert_precision = None
    bert_recall = None
    bert_f1 = None
    try:
        from bert_score import score as bert_score_fn

        preds = [e.get("prediction", "") for e in entries]
        refs = [e["ground_truth"] for e in entries]
        P, R, F1 = bert_score_fn(preds, refs, lang="en", verbose=False, batch_size=64)
        bert_precision = P.mean().item() * 100
        bert_recall = R.mean().item() * 100
        bert_f1 = F1.mean().item() * 100
    except ImportError:
        pass
    except Exception as e:
        print(f"BERTScore error: {e}")

    result = {
        "total": total,
        "exact_match": exact / total * 100 if total else 0,
        "containment_match": contain / total * 100 if total else 0,
    }

    if bert_f1 is not None:
        result["bertscore_precision"] = round(bert_precision, 2)
        result["bertscore_recall"] = round(bert_recall, 2)
        result["bertscore_f1"] = round(bert_f1, 2)

    return result


SCORERS = {
    "fitzpatrick17k": score_fitzpatrick,
    "mm_skin_vqa": score_vqa,
    "confusion_triads": score_confusion_triads,
}


def main():
    parser = argparse.ArgumentParser(description="Score benchmark results")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Score specific model")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODELS.keys())

    all_results = {}
    for model_key in models:
        model_dir = RESULTS_DIR / model_key
        if not model_dir.exists():
            continue

        all_results[model_key] = {}
        for bench_key, scorer in SCORERS.items():
            results_file = model_dir / bench_key / "predictions.jsonl"
            if results_file.exists():
                scores = scorer(results_file)
                all_results[model_key][bench_key] = scores
                print(f"\n{'='*60}")
                print(f"{model_key} | {bench_key}")
                print(f"{'='*60}")
                print(json.dumps(scores, indent=2))

    # Generate comparison table
    if all_results:
        print(f"\n{'='*60}")
        print("COMPARISON TABLE")
        print(f"{'='*60}")
        print(f"{'Model':20s} | {'Fitz Top-1':>10s} | {'Fitz Top-6':>10s} | {'Triads':>8s} | {'VQA Match':>10s}")
        print("-" * 70)
        for model_key, benchmarks in all_results.items():
            fitz = benchmarks.get("fitzpatrick17k", {})
            triads = benchmarks.get("confusion_triads", {})
            vqa = benchmarks.get("mm_skin_vqa", {})
            print(
                f"{model_key:20s} | "
                f"{fitz.get('top1_accuracy', '-'):>9.1f}% | "
                f"{fitz.get('top6_accuracy', '-'):>9.1f}% | "
                f"{triads.get('accuracy', '-'):>6.1f}% | "
                f"{vqa.get('containment_match', '-'):>9.1f}%"
            )

    # Save combined results
    output = RESULTS_DIR / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
