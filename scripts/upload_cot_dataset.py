"""Reformat ``danielfdias98/derm-reasoning-full-reasoning`` into a chain-of-thought
(CoT) variant for the dissertation Run-9 format-ablation experiment.

The transformation is minimal-disruption: each row's ``response`` field is split
into a reasoning-first sequence inspired by MedVLM-R1 (Pan et al., arXiv:2502.19634)::

    <think>
    {original `reasoning` text}
    </think>
    <answer>
    {original JSON with the `reasoning` key removed}
    </answer>

All other fields (observation, morphology, colour, texture, border, distribution,
diagnosis, category) remain inside the ``<answer>`` JSON so the supervision signal
on the structured fields is preserved. Image, instruction, image_id, class, and
source fields are passed through unchanged.

Why minimal-disruption
======================

We want Run 9 vs Run 8 (same model, same hyperparameters, same content) to
attribute purely to **reasoning-before-answer vs answer-before-reasoning**.
Rewriting the reasoning prose itself with a different teacher would introduce
a second confound (teacher quality). See implementation_log.md §11.1 + §11.4.

Usage
=====

::

    # Dry run: print N sample transformations and exit (no token needed)
    python scripts/upload_cot_dataset.py --dry-run --samples 3

    # Full push (after dry-run verification)
    HF_TOKEN=hf_... python scripts/upload_cot_dataset.py --push
"""
from __future__ import annotations

import argparse
import json
import os
import sys

SOURCE_REPO = "danielfdias98/derm-reasoning-full-reasoning"
DEST_REPO = "danielfdias98/derm-reasoning-cot"


def transform_response(response_field) -> str:
    """Convert a structured-JSON response to the ``<think>/<answer>`` CoT format.

    Accepts either a JSON-encoded string or a dict and returns the formatted
    string. If parsing fails, the original payload is wrapped in ``<answer>``
    with an empty ``<think>`` block (so the row is still well-formed).
    """
    if isinstance(response_field, str):
        try:
            d = json.loads(response_field)
        except json.JSONDecodeError:
            return f"<think>\n\n</think>\n<answer>\n{response_field}\n</answer>"
    elif isinstance(response_field, dict):
        d = dict(response_field)
    else:
        return f"<think>\n\n</think>\n<answer>\n{str(response_field)}\n</answer>"

    reasoning = (d.pop("reasoning", "") or "").strip()
    answer_json = json.dumps(d, ensure_ascii=False)
    return f"<think>\n{reasoning}\n</think>\n<answer>\n{answer_json}\n</answer>"


def to_cot_row(row: dict) -> dict:
    """Map a single row to its CoT-formatted version (other fields untouched)."""
    row = dict(row)
    row["response"] = transform_response(row["response"])
    return row


def _pretty(payload, max_chars: int = 2000) -> str:
    """Pretty-print a JSON-ish payload, truncated to ``max_chars``."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            s = payload
            return s if len(s) <= max_chars else s[:max_chars] + "\n  ... [truncated]"
    s = json.dumps(payload, indent=2, ensure_ascii=False)
    return s if len(s) <= max_chars else s[:max_chars] + "\n  ... [truncated]"


def dry_run(n_samples: int, source_repo: str) -> None:
    from datasets import load_dataset
    import random

    print(f"Loading source: {source_repo}")
    ds = load_dataset(source_repo, split="train")
    print(f"  train rows: {len(ds):,}")

    random.seed(7)
    indices = random.sample(range(len(ds)), n_samples)
    print(f"  sampling indices (seed=7): {indices}")

    for i, idx in enumerate(indices, 1):
        row = ds[idx]
        new_row = to_cot_row(row)
        print(f"\n{'=' * 72}")
        print(f"SAMPLE {i}/{n_samples}  (row index {idx})")
        print(
            f"  image_id={row['image_id']}  "
            f"class={row['class']}  source={row['source']}"
        )
        print("\n--- BEFORE response (pretty-printed JSON) ---")
        print(_pretty(row["response"]))
        print("\n--- AFTER response (raw, what the model will see) ---")
        print(new_row["response"][:2000] + (
            "\n  ... [truncated]" if len(new_row["response"]) > 2000 else ""
        ))
        print()


def push(source_repo: str, dest_repo: str) -> None:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN env var not set. Export it before --push.")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=dest_repo, repo_type="dataset", private=False, exist_ok=True
    )

    print(f"Source: {source_repo}")
    src = load_dataset(source_repo)
    for split, sub in src.items():
        print(f"  {split}: {len(sub):,} rows")

    print("\nApplying CoT transform via .map() ...")
    out = src.map(to_cot_row, desc="cot-transform")
    print("Pushing to hub ...")
    out.push_to_hub(dest_repo, token=token)
    print(f"\nPushed → https://huggingface.co/datasets/{dest_repo}")
    upload_card(dest_repo)


def upload_card(repo_id: str) -> None:
    from huggingface_hub import HfApi

    body = f"""---
language: [en]
license: cc-by-nc-sa-4.0
task_categories: [image-text-to-text, visual-question-answering]
size_categories: [10K<n<100K]
tags: [medical, dermatology, vision-language, vlm-finetune, fairness, chain-of-thought]
pretty_name: Dermatology Reasoning Dataset — Chain-of-Thought Variant
---

# Dermatology Reasoning Dataset — Chain-of-Thought (CoT) Variant

This is the **chain-of-thought formatted** variant of
[danielfdias98/derm-reasoning-full-reasoning](https://huggingface.co/datasets/danielfdias98/derm-reasoning-full-reasoning).
The image set, the per-row clinical content, and the diagnostic labels are
identical to the full-reasoning repo; only the **assistant-turn formatting**
differs.

## Format

Every row's assistant turn follows the MedVLM-R1 convention
(Pan et al., *MedVLM-R1: Incentivizing Medical Reasoning Capability of
Vision-Language Models via Reinforcement Learning*, arXiv:2502.19634, 2025):

```
<think>
[free-form clinical reasoning narrative]
</think>
<answer>
{{"diagnosis": "...", "category": "...", "morphology": "...", ...}}
</answer>
```

The `reasoning` text appears **before** the structured answer, encouraging
chain-of-thought generation prior to label commitment. All other structured
fields (`observation`, `morphology`, `colour`, `texture`, `border`,
`distribution`, `diagnosis`, `category`) are preserved inside the `<answer>`
JSON, so the rich per-field supervision signal is unchanged.

## Why this variant exists

The dissertation's main 8-run campaign uses the JSON-first format
(`derm-reasoning-full-reasoning`). This CoT variant supports a single
methodological-extension run (Run 9 in implementation_log.md §11.1) on Qwen
3.5 9B, isolating the contribution of response-format ordering from the
contributions of supervision density and architecture (which are tested by
the main factorial). All hyperparameters except dataset are held constant
between Run 8 (JSON format) and Run 9 (CoT format).

## Quick load

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}")
print(ds)  # DatasetDict {{train: 25_637, val: 2_849}}
```

## Relationship to sibling repos

| Variant | Format | Used in |
|---|---|---|
| [`danielfdias98/derm-reasoning-label-only`](https://huggingface.co/datasets/danielfdias98/derm-reasoning-label-only) | Diagnosis + category labels only | Runs 1, 3, 5, 7 (label-only baselines) |
| [`danielfdias98/derm-reasoning-full-reasoning`](https://huggingface.co/datasets/danielfdias98/derm-reasoning-full-reasoning) | JSON-first structured | Runs 2, 4, 6, 8 (full-reasoning) |
| **`{repo_id}`** | Reasoning-first ``<think>``/``<answer>`` | Run 9 (format ablation on Qwen 3.5 9B) |

## License

CC-BY-NC-SA 4.0 (matches the most restrictive component of the source
datasets, SkinCAP). Non-commercial research use only.

## Citation

```bibtex
@misc{{dias2026derm-reasoning-cot,
  author = {{Ferreira Dias, Daniel}},
  title  = {{Dermatology Reasoning Dataset — Chain-of-Thought Variant}},
  year   = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}},
}}
```
"""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=body.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="README for CoT variant",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample transformations and exit (no token needed)",
    )
    g.add_argument(
        "--push",
        action="store_true",
        help="Push transformed dataset to HF Hub (HF_TOKEN required)",
    )
    p.add_argument(
        "--samples", type=int, default=3, help="Dry-run sample count (default 3)"
    )
    p.add_argument("--source-repo", default=SOURCE_REPO)
    p.add_argument("--dest-repo", default=DEST_REPO)
    args = p.parse_args()

    if args.dry_run:
        dry_run(args.samples, args.source_repo)
    else:
        push(args.source_repo, args.dest_repo)


if __name__ == "__main__":
    main()
