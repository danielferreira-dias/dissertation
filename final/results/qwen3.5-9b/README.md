# Qwen 3.5 9B — Zero-Shot Baseline Results (v2)

- **Model:** `Qwen/Qwen3.5-9B` (9B parameters)
- **Type:** General-purpose natively multimodal VLM (early fusion)
- **Evaluation:** Zero-shot (no fine-tuning), vLLM guided JSON structured output
- **Infrastructure:** NVIDIA L40S (48GB), vLLM 0.19.0, bfloat16, batch size 32
- **Date:** 2026-04-09

---

## Fitzpatrick17k Benchmark (1,000 images, 114 conditions)

| Metric | Value |
|--------|-------|
| **Top-1 accuracy** | **17.3%** |
| **Top-6 accuracy** | **35.0%** |

### Fairness by Fitzpatrick Skin Type

| FST | Total | Top-1 Accuracy |
|-----|-------|---------------|
| I | 207 | 15.9% |
| II | 296 | 15.9% |
| III | 197 | 17.8% |
| IV | 170 | 20.0% |
| V | 94 | 19.1% |
| VI | 36 | 16.7% |

**Good fairness profile** — only 4.1% spread between best and worst FST. Slightly better on darker skin (FST IV-V) than lighter skin.

### Comparison with published baselines

| Model | Parameters | Top-1 | Top-6 | Source |
|-------|-----------|-------|-------|--------|
| SkinFlow (fine-tuned) | 7B | 29.19% | 71.16% | Liu et al., 2026 |
| MedGemma 4B (zero-shot) | 4B | 19.7% | 31.1% | This work |
| GPT-5.2 | Commercial | 18.24% | 42.59% | Liu et al., 2026 |
| **Qwen 3.5 9B (zero-shot)** | **9B** | **17.3%** | **35.0%** | **This work** |
| Qwen3-VL-235B | 235B | 17.13% | — | Liu et al., 2026 |

**Matches Qwen3-VL-235B (17.3% vs 17.13%) at 26x fewer parameters. Highest Top-6 accuracy among our models at 35.0%.**

---

## Confusion Triads (820 images, 6 classes)

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **67.1%** (550/820) |

### Per-class breakdown

| Class | Correct | Total | Accuracy | Confusion Zone |
|-------|---------|-------|----------|---------------|
| Seborrheic Keratosis | 127 | 150 | **85%** | Lesion triad |
| Seborrheic Dermatitis | 55 | 70 | **79%** | Inflammatory triad |
| Basal Cell Carcinoma | 114 | 150 | **76%** | Lesion triad |
| Psoriasis | 97 | 150 | **65%** | Inflammatory triad |
| Eczema | 92 | 150 | **61%** | Inflammatory triad |
| Melanoma | 65 | 150 | **43%** | Lesion triad |

### Key findings

1. **Best overall triads accuracy (67.1%)** across all models tested.
2. **Melanoma at 43% — best zero-shot melanoma detection.** Still inadequate clinically but a significant improvement over MedGemma (3%) and Qwen 4B (15%).
3. **Balanced performance across all 6 classes** — no single class below 43%. Most equitable confusion triad results.
4. **BCC at 76%** — strong malignant lesion detection, highest of all models.
5. **Seb dermatitis at 79%** — excellent for our focus condition.

---

## MM-Skin VQA (5,452 QA pairs)

| Metric | Value |
|--------|-------|
| **BERTScore F1** | **88.36%** |
| BERTScore Precision | 87.17% |
| BERTScore Recall | 89.65% |
| Exact match | 0.1% |
| Containment match | 0.1% |

---

## Files

- `fitzpatrick17k/predictions.jsonl` — 1,000 predictions (guided JSON)
- `mm_skin_vqa/predictions.jsonl` — 5,452 VQA predictions
- `confusion_triads/predictions.jsonl` — 820 triad predictions
