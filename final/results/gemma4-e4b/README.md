# Gemma 4 E4B IT — Zero-Shot Baseline Results (v2)

- **Model:** `google/gemma-4-E4B-it` (~4B parameters)
- **Type:** General-purpose VLM (no medical pre-training)
- **Evaluation:** Zero-shot (no fine-tuning), vLLM guided JSON structured output
- **Infrastructure:** NVIDIA L40S (48GB), vLLM 0.19.0, bfloat16, batch size 32
- **Date:** 2026-04-09

---

## Fitzpatrick17k Benchmark (1,000 images, 114 conditions)

| Metric | Value |
|--------|-------|
| **Top-1 accuracy** | **5.9%** |
| **Top-6 accuracy** | **22.1%** |

### Fairness by Fitzpatrick Skin Type

| FST | Total | Top-1 Accuracy |
|-----|-------|---------------|
| I | 207 | 5.3% |
| II | 296 | 8.1% |
| III | 197 | 6.1% |
| IV | 170 | 7.1% |
| V | 94 | 0.0% |
| VI | 36 | 0.0% |

**Critical fairness gap:** 0% accuracy on FST V-VI (dark skin). The model completely fails on darker skin tones.

### Comparison with published baselines

| Model | Parameters | Top-1 | Top-6 | Source |
|-------|-----------|-------|-------|--------|
| SkinFlow (fine-tuned) | 7B | 29.19% | 71.16% | Liu et al., 2026 |
| GPT-5.2 | Commercial | 18.24% | 42.59% | Liu et al., 2026 |
| **Gemma 4 E4B (zero-shot)** | **~4B** | **5.9%** | **22.1%** | **This work** |

---

## Confusion Triads (820 images, 6 classes)

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **45.4%** (372/820) |

### Per-class breakdown

| Class | Correct | Total | Accuracy | Confusion Zone |
|-------|---------|-------|----------|---------------|
| Seborrheic Keratosis | 114 | 150 | **76%** | Lesion triad |
| Eczema | 111 | 150 | **74%** | Inflammatory triad |
| Seborrheic Dermatitis | 52 | 70 | **74%** | Inflammatory triad |
| Basal Cell Carcinoma | 36 | 150 | **24%** | Lesion triad |
| Melanoma | 36 | 150 | **24%** | Lesion triad |
| Psoriasis | 23 | 150 | **15%** | Inflammatory triad |

### Key findings

1. **Eczema and seb dermatitis are strong (74%)** — better than MedGemma on these conditions.
2. **Psoriasis at 15%** reveals a critical blind spot without medical training.
3. **BCC and melanoma both at 24%** — cannot distinguish malignant lesions.

---

## MM-Skin VQA (5,452 QA pairs)

| Metric | Value |
|--------|-------|
| **BERTScore F1** | **88.64%** |
| BERTScore Precision | 87.69% |
| BERTScore Recall | 89.65% |
| Exact match | 0.0% |
| Containment match | 0.0% |

---

## Files

- `fitzpatrick17k/predictions.jsonl` — 1,000 predictions (guided JSON)
- `mm_skin_vqa/predictions.jsonl` — 5,452 VQA predictions
- `confusion_triads/predictions.jsonl` — 820 triad predictions
