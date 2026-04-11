# Gemma 4 E4B IT — Zero-Shot Baseline Results (v2)

- **Model:** `google/gemma-4-E4B-it` (~4B parameters)
- **Type:** General-purpose VLM (no medical pre-training)
- **Evaluation:** Zero-shot (no fine-tuning), vLLM guided JSON structured output
- **Infrastructure:** NVIDIA L40S (48GB), vLLM 0.19.0, bfloat16, batch size 16
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

### Comparison with published baselines

| Model | Parameters | Top-1 | Top-6 | Source |
|-------|-----------|-------|-------|--------|
| SkinFlow (fine-tuned) | 7B | 29.19% | 71.16% | Liu et al., 2026 |
| GPT-5.2 | Commercial | 18.24% | 42.59% | Liu et al., 2026 |
| Qwen3-VL-235B | 235B | 17.13% | — | Liu et al., 2026 |
| **Gemma 4 E4B IT (zero-shot)** | **~4B** | **5.9%** | **22.1%** | **This work** |

---

## Confusion Triads (820 images, 6 classes)

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **45.4%** (372/820) |

### Per-class breakdown

| Class | Correct | Total | Accuracy | Confusion Zone |
|-------|---------|-------|----------|---------------|
| Seborrheic Keratosis | 114 | 150 | **76%** | Lesion triad |
| Seborrheic Dermatitis | 52 | 70 | **74%** | Inflammatory triad |
| Eczema | 111 | 150 | **74%** | Inflammatory triad |
| Basal Cell Carcinoma | 36 | 150 | **24%** | Lesion triad |
| Melanoma | 36 | 150 | **24%** | Lesion triad |
| Psoriasis | 23 | 150 | **15%** | Inflammatory triad |

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
