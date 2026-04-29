---
language: [en]
license: cc-by-nc-sa-4.0
task_categories: [image-text-to-text, visual-question-answering]
size_categories: [10K<n<100K]
tags: [medical, dermatology, vision-language, vlm-finetune, fairness, chain-of-thought, dataset-cleaning]
pretty_name: Dermatology Reasoning Dataset — Visible Thinking v2
---

# Dermatology Reasoning Dataset — Visible Thinking v2

This is a cleaned visible-thinking successor to
[`danielfdias98/derm-reasoning-full-reasoning`](https://huggingface.co/datasets/danielfdias98/derm-reasoning-full-reasoning).
The original dataset and the earlier CoT variant are not overwritten.

Every assistant response is formatted as:

```text
<think>
Visible evidence:
- ...

Differential reasoning:
- ...

Limitations:
- Symptoms, duration, tenderness, palpation findings, lab confirmation, dermoscopy, and patient history are not inferable from the image alone unless explicitly visible or provided.
</think>
<answer>
{"diagnosis": "...", "category": "...", "confidence": "...", ...}
</answer>
```

The `<think>` block is rebuilt deterministically from visible structured fields
(`morphology`, `color`, `texture`, `border`, `distribution`, and `observation`).
The original free-form `reasoning` field is **not copied** into the thinking
block.

## Cleaning Summary

| Metric | Count |
|---|---:|
| Source rows | 28486 |
| Train rows kept | 22528 |
| Validation rows kept | 2539 |
| Rows quarantined | 3419 |
| Duplicate hash groups detected | 506 |
| Post-clean train/val duplicate hashes | 0 |

## Quarantine Reasons

| Reason | Rows |
|---|---:|
| `duplicate_conflicting_class` | 330 |
| `duplicate_cross_split_train_leak` | 78 |
| `duplicate_same_split_noncanonical` | 260 |
| `hard_dermoscopy` | 744 |
| `hard_nonclinical_diagnostic_media` | 298 |
| `low_confidence` | 1827 |

Rows were quarantined, not silently discarded during auditing. The build script
preserves `audit/quarantine.jsonl`, `audit/duplicate_groups.jsonl`, and
`audit/summary.json` locally for reproducibility.

## Schema

- `image`: embedded image
- `instruction`: user instruction
- `response`: `<think>/<answer>` formatted assistant response
- `image_id`: filename stem
- `class`: original class folder label
- `source`: attributed source dataset
- `content_hash`: SHA-256 of image bytes
- `quality_flags`: deterministic cleanup flags

## Quick Load

```python
from datasets import load_dataset

ds = load_dataset("danielfdias98/derm-reasoning-think-v2")
print(ds)
print(ds["train"][0]["response"])
```

## License

CC-BY-NC-SA 4.0, inheriting the most restrictive component of the source data.
Non-commercial research use only.

## Citation

```bibtex
@misc{dias2026derm-reasoning-think-v2,
  author = {Ferreira Dias, Daniel},
  title  = {Dermatology Reasoning Dataset — Visible Thinking v2},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/datasets/danielfdias98/derm-reasoning-think-v2}},
}
```
