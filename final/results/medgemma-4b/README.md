# MedGemma 1.5 4B IT — Zero-Shot Baseline Results (v2)

- **Model:** `google/medgemma-1.5-4b-it` (4B parameters)
- **Type:** Medical-specialized VLM (pre-trained on dermatology, radiology, pathology)
- **Evaluation:** Zero-shot (no fine-tuning), vLLM guided JSON structured output
- **Infrastructure:** NVIDIA L40S (48GB), vLLM 0.19.0, bfloat16, batch size 32
- **Date:** 2026-04-09

---

## Fitzpatrick17k Benchmark (1,000 images, 114 conditions)

| Metric | Value |
|--------|-------|
| **Top-1 accuracy** | **19.7%** |
| **Top-6 accuracy** | **31.1%** |

### Fairness by Fitzpatrick Skin Type

| FST | Total | Top-1 Accuracy |
|-----|-------|---------------|
| I | 207 | 22.7% |
| II | 296 | 20.6% |
| III | 197 | 12.7% |
| IV | 170 | 21.2% |
| V | 94 | 20.2% |
| VI | 36 | 25.0% |

### Comparison with published baselines

| Model | Parameters | Top-1 | Top-6 | Source |
|-------|-----------|-------|-------|--------|
| SkinFlow (fine-tuned) | 7B | 29.19% | 71.16% | Liu et al., 2026 |
| **MedGemma 4B (zero-shot)** | **4B** | **19.7%** | **31.1%** | **This work** |
| GPT-5.2 | Commercial | 18.24% | 42.59% | Liu et al., 2026 |
| Qwen3-VL-235B | 235B | 17.13% | — | Liu et al., 2026 |

**Exceeds GPT-5.2 on Top-1 (19.7% vs 18.24%) — a 4B medical model outperforms a commercial LLM.**

---

## Confusion Triads (820 images, 6 classes)

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **53.3%** (437/820) |

### Per-class breakdown

| Class | Correct | Total | Accuracy | Confusion Zone |
|-------|---------|-------|----------|---------------|
| Seborrheic Keratosis | 137 | 150 | **91%** | Lesion triad |
| Basal Cell Carcinoma | 103 | 150 | **69%** | Lesion triad |
| Psoriasis | 82 | 150 | **55%** | Inflammatory triad |
| Seborrheic Dermatitis | 36 | 70 | **51%** | Inflammatory triad |
| Eczema | 74 | 150 | **49%** | Inflammatory triad |
| Melanoma | 5 | 150 | **3%** | Lesion triad |

### Key findings

1. **Melanoma at 3% is a critical safety failure.** The most life-threatening condition is the worst detected. Melanoma is overwhelmingly misclassified as seborrheic keratosis (118/150).
2. **Seborrheic keratosis at 91%** — the model's strongest class, but it over-predicts this condition across the board.
3. **BCC improved to 69%** with structured output forcing focused diagnosis.

---

## MM-Skin VQA (5,452 QA pairs)

| Metric | Value |
|--------|-------|
| **BERTScore F1** | **89.36%** |
| BERTScore Precision | 88.85% |
| BERTScore Recall | 89.92% |
| Exact match | 0.2% |
| Containment match | 0.2% |

**Highest BERTScore F1 of all models tested** — medical pre-training helps with semantic answer quality.

---

## Files

- `fitzpatrick17k/predictions.jsonl` — 1,000 predictions (guided JSON)
- `mm_skin_vqa/predictions.jsonl` — 5,452 VQA predictions
- `confusion_triads/predictions.jsonl` — 820 triad predictions
