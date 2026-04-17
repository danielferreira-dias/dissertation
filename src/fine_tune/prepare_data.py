"""Convert reasoning.jsonl into stratified train/val splits in chat format.

Emits BOTH format variants from the same stratified split for an ablation
comparison:
  - <split>.label_only.jsonl     — assistant = {diagnosis, category}
  - <split>.full_reasoning.jsonl — assistant = unified schema per spec §3.2

Run once per dataset. Output consumed by `src/fine_tune/train.py`.

Usage:
    python -m fine_tune.prepare_data \
        --source data/reasoning/reasoning.jsonl \
        --out-dir data/reasoning \
        --image-root /workspace/data/images \
        --val-fraction 0.05 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


USER_PROMPT = "Diagnose this skin condition with structured reasoning."


def _prettify(label: str) -> str:
    return label.replace("_", " ").title()


def build_label_only_payload(entry: dict[str, Any]) -> dict[str, Any]:
    """Minimal assistant response: diagnosis + category only."""
    # Anchor to ground truth (not reasoner's diagnosis) — the reasoner may
    # have defended a wrong label; label_match=false entries are filtered elsewhere.
    return {
        "diagnosis": _prettify(entry["ground_truth"]),
        "category": entry.get("category", ""),
    }


def build_full_reasoning_payload(entry: dict[str, Any]) -> dict[str, Any]:
    """Full unified schema per spec §3.2."""
    reasoning = entry.get("reasoning", {}) or {}
    observation = entry.get("observation", {}) or {}
    # Anchor to ground truth (not reasoner's diagnosis) — the reasoner may
    # have defended a wrong label; label_match=false entries are filtered elsewhere.
    diagnosis = _prettify(entry["ground_truth"])
    differentials = reasoning.get("differentials") or []

    top_n = [diagnosis] + [
        d.get("condition", "").strip().title()
        for d in differentials
        if d.get("condition", "").strip()
    ]

    return {
        "diagnosis": diagnosis,
        "top_n": top_n,
        "confidence": reasoning.get("confidence", "medium"),
        "category": entry.get("category", ""),
        "observation": {
            "morphology": observation.get("morphology", ""),
            "color": observation.get("color", ""),
            "texture": observation.get("texture", ""),
            "border": observation.get("border", ""),
            "distribution": observation.get("distribution", ""),
        },
        "reasoning": reasoning.get("clinical_reasoning", ""),
        "differentials": differentials,
    }


_BUILDERS: dict[str, Callable[[dict], dict]] = {
    "label_only": build_label_only_payload,
    "full_reasoning": build_full_reasoning_payload,
}

FORMATS = tuple(_BUILDERS.keys())


def entry_to_chat(entry: dict[str, Any], image_root: Path, format_name: str) -> dict[str, Any]:
    """Build a single chat-format training example for the given format."""
    if format_name not in _BUILDERS:
        raise ValueError(f"unknown format: {format_name!r}; expected one of {list(_BUILDERS)}")
    abs_image = str((image_root / entry["image_path"]).as_posix())
    payload = _BUILDERS[format_name](entry)
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": abs_image},
                {"type": "text", "text": USER_PROMPT},
            ]},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ]
    }


def prepare_splits(
    source: Path,
    out_dir: Path,
    val_fraction: float,
    image_root: Path,
    seed: int,
) -> dict[str, int]:
    """Load reasoning.jsonl, filter flagged, stratified-split, emit both formats.

    Writes 5 files into out_dir: train/val × label_only/full_reasoning + audit.jsonl.
    Returns a summary dict with counts.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    with Path(source).open() as f:
        entries = [json.loads(l) for l in f if l.strip()]

    audited: list[dict] = []
    kept: list[dict] = []
    for e in entries:
        if (e.get("reasoning") or {}).get("label_match") is False:
            audited.append(e)
        else:
            kept.append(e)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for e in kept:
        by_class[e["ground_truth"]].append(e)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for label in sorted(by_class.keys()):
        bucket = by_class[label][:]
        rng.shuffle(bucket)
        n_val = max(1, int(round(len(bucket) * val_fraction))) if len(bucket) >= 2 else 0
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for format_name in FORMATS:
        for split_name, split_entries in (("train", train), ("val", val)):
            path = out_dir / f"{split_name}.{format_name}.jsonl"
            with path.open("w") as f:
                for e in split_entries:
                    f.write(json.dumps(entry_to_chat(e, image_root, format_name)) + "\n")

    audit_path = out_dir / "audit.jsonl"
    with audit_path.open("w") as f:
        for e in audited:
            f.write(json.dumps(e) + "\n")

    return {
        "total": len(entries),
        "audited": len(audited),
        "train": len(train),
        "val": len(val),
        "classes": len(by_class),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/reasoning/reasoning.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/reasoning"))
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--image-root", type=Path, default=Path("/workspace/data/images"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    stats = prepare_splits(
        source=args.source,
        out_dir=args.out_dir,
        val_fraction=args.val_fraction,
        image_root=args.image_root,
        seed=args.seed,
    )
    print(f"Source entries:     {stats['total']}")
    print(f"Audited (flagged):  {stats['audited']}")
    print(f"Train examples:     {stats['train']}  (per format)")
    print(f"Val examples:       {stats['val']}  (per format)")
    print(f"Classes:            {stats['classes']}")
    print("Emitted:")
    for fmt in FORMATS:
        for split in ("train", "val"):
            print(f"  {args.out_dir / f'{split}.{fmt}.jsonl'}")


if __name__ == "__main__":
    main()
