# MedGemma 1.5 4B IT — Zero-Shot Baseline Results

- **Model:** `google/medgemma-1.5-4b-it` (4B parameters)
- **Type:** Medical-specialized VLM (pre-trained on dermatology, radiology, pathology)
- **Evaluation:** Zero-shot (no fine-tuning)
- **Infrastructure:** NVIDIA L40S (48GB), vLLM 0.19.0, bfloat16, batch size 16
- **Date:** 2026-04-08

---

## Fitzpatrick17k Benchmark (1,000 images, 114 conditions)

| Metric | Value |
|--------|-------|
| **Top-1 accuracy** | **16.3%** |
| JSON format compliance | 80% (800/1000) |

### Comparison with published baselines

| Model | Parameters | Top-1 | Source |
|-------|-----------|-------|--------|
| SkinFlow (fine-tuned) | 7B | 29.19% | Liu et al., 2026 |
| GPT-5.2 | N/A | 18.24% | Liu et al., 2026 |
| Qwen3-VL-235B | 235B | 17.13% | Liu et al., 2026 |
| **MedGemma 4B (zero-shot)** | **4B** | **16.3%** | **This work** |

**Gap to close:** +1.94% to match GPT-5.2, +12.89% to match SkinFlow.

---

## MM-Skin VQA Benchmark (5,452 QA pairs)

| Metric | Value |
|--------|-------|
| Predictions completed | 5,452/5,452 |
| Scoring | Pending — BERTScore to be applied after all models complete |

---

## Confusion Triads (820 images, 6 classes)

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **54.5%** (447/820) |

### Per-class breakdown

| Class | Correct | Total | Accuracy | Confusion Zone |
|-------|---------|-------|----------|---------------|
| Seborrheic Keratosis | 137 | 150 | **91%** | Lesion triad |
| Seborrheic Dermatitis | 44 | 70 | **63%** | Inflammatory triad |
| Psoriasis | 93 | 150 | **62%** | Inflammatory triad |
| Basal Cell Carcinoma | 83 | 150 | **55%** | Lesion triad |
| Eczema | 79 | 150 | **53%** | Inflammatory triad |
| Melanoma | 11 | 150 | **7%** | Lesion triad |

### Key findings

1. **Melanoma at 7% is a critical safety failure.** The most life-threatening condition is the worst detected. This validates the need for fine-tuning with structured reasoning — zero-shot models cannot be trusted for melanoma detection.

2. **Inflammatory triad (53-63%):** Partial differentiation between seb dermatitis, psoriasis, and eczema. The model recognizes some distinguishing features but confuses conditions that share erythematous, scaly presentations.

3. **Lesion triad asymmetry (7-91%):** Strong recognition of benign seb keratosis but catastrophic failure on melanoma. Likely reflects training data bias — benign lesions vastly outnumber melanomas in general training corpora.

4. **Seborrheic dermatitis at 63%:** Reasonable baseline for our focus condition.

---

## Files

- `fitzpatrick17k/predictions.jsonl` — 1,000 predictions with ground truth, raw response, parsed JSON
- `mm_skin_vqa/predictions.jsonl` — 5,452 VQA predictions
- `confusion_triads/predictions.jsonl` — 820 triad predictions

## Pending

- BERTScore for VQA semantic similarity
- Fairness analysis per Fitzpatrick skin type
- Confusion matrix (6x6) for triads
