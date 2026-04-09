# Qwen 3.5 4B — Zero-Shot Baseline Results (v2)

- **Model:** `Qwen/Qwen3.5-4B` (4B parameters)
- **Type:** General-purpose natively multimodal VLM (early fusion)
- **Evaluation:** Zero-shot (no fine-tuning), vLLM guided JSON structured output
- **Infrastructure:** NVIDIA L40S (48GB), vLLM 0.19.0, bfloat16, batch size 32
- **Date:** 2026-04-09

---

## Fitzpatrick17k Benchmark (1,000 images, 114 conditions)

| Metric | Value |
|--------|-------|
| **Top-1 accuracy** | **12.0%** |
| **Top-6 accuracy** | **27.7%** |

### Fairness by Fitzpatrick Skin Type

| FST | Total | Top-1 Accuracy |
|-----|-------|---------------|
| I | 207 | 11.6% |
| II | 296 | 9.5% |
| III | 197 | 13.2% |
| IV | 170 | 13.5% |
| V | 94 | 14.9% |
| VI | 36 | 13.9% |

**Most equitable model across skin tones** — only 5.4% spread between best and worst FST (vs 25% for Gemma 4). Slightly better on darker skin (FST V-VI) than lighter skin.

### Comparison with published baselines

| Model | Parameters | Top-1 | Top-6 | Source |
|-------|-----------|-------|-------|--------|
| SkinFlow (fine-tuned) | 7B | 29.19% | 71.16% | Liu et al., 2026 |
| GPT-5.2 | Commercial | 18.24% | 42.59% | Liu et al., 2026 |
| Qwen3-VL-235B | 235B | 17.13% | — | Liu et al., 2026 |
| **Qwen 3.5 4B (zero-shot)** | **4B** | **12.0%** | **27.7%** | **This work** |

---

## Confusion Triads (820 images, 6 classes)

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **53.0%** (435/820) |

### Per-class breakdown

| Class | Correct | Total | Accuracy | Confusion Zone |
|-------|---------|-------|----------|---------------|
| Seborrheic Keratosis | 135 | 150 | **90%** | Lesion triad |
| Basal Cell Carcinoma | 104 | 150 | **69%** | Lesion triad |
| Psoriasis | 85 | 150 | **57%** | Inflammatory triad |
| Seborrheic Dermatitis | 39 | 70 | **56%** | Inflammatory triad |
| Eczema | 49 | 150 | **33%** | Inflammatory triad |
| Melanoma | 23 | 150 | **15%** | Lesion triad |

### Key findings

1. **Seborrheic keratosis at 90%** — strong benign lesion recognition.
2. **BCC at 69%** — good malignant lesion detection for a 4B model.
3. **Melanoma at 15%** — misclassified primarily as seborrheic keratosis (104/150). Critical fine-tuning target.
4. **Eczema drops to 33%** under structured output — the model struggles to differentiate eczema from psoriasis and seb keratosis when forced to commit to a single diagnosis.

---

## MM-Skin VQA (5,452 QA pairs)

| Metric | Value |
|--------|-------|
| **BERTScore F1** | **88.39%** |
| BERTScore Precision | 87.16% |
| BERTScore Recall | 89.71% |
| Exact match | 0.2% |
| Containment match | 0.2% |

---

## Files

- `fitzpatrick17k/predictions.jsonl` — 1,000 predictions (guided JSON)
- `mm_skin_vqa/predictions.jsonl` — 5,452 VQA predictions
- `confusion_triads/predictions.jsonl` — 820 triad predictions
