# Gemma 4 E4B IT — Zero-Shot Baseline Results

- **Model:** `google/gemma-4-E4B-it` (~4B parameters)
- **Type:** General-purpose VLM (no medical pre-training)
- **Evaluation:** Zero-shot (no fine-tuning)
- **Infrastructure:** NVIDIA L40S (48GB), vLLM 0.19.0, bfloat16, batch size 16
- **Date:** 2026-04-08

---

## Fitzpatrick17k Benchmark (1,000 images, 114 conditions)

| Metric | Value |
|--------|-------|
| **Top-1 accuracy** | **19.0%** |

### Comparison with published baselines

| Model | Parameters | Top-1 | Source |
|-------|-----------|-------|--------|
| SkinFlow (fine-tuned) | 7B | 29.19% | Liu et al., 2026 |
| **Gemma 4 E4B (zero-shot)** | **~4B** | **19.0%** | **This work** |
| GPT-5.2 | N/A | 18.24% | Liu et al., 2026 |
| MedGemma 4B (zero-shot) | 4B | 16.3% | This work |
| Qwen3-VL-235B | 235B | 17.13% | Liu et al., 2026 |

**Exceeds GPT-5.2 zero-shot** — a 4B general-purpose model outperforms a massive commercial LLM.

---

## Confusion Triads (820 images, 6 classes)

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **46.1%** (378/820) |

### Per-class breakdown

| Class | Correct | Total | Accuracy | Confusion Zone |
|-------|---------|-------|----------|---------------|
| Eczema | 127 | 150 | **85%** | Inflammatory triad |
| Seborrheic Keratosis | 107 | 150 | **71%** | Lesion triad |
| Seborrheic Dermatitis | 39 | 70 | **56%** | Inflammatory triad |
| Basal Cell Carcinoma | 47 | 150 | **31%** | Lesion triad |
| Melanoma | 42 | 150 | **28%** | Lesion triad |
| Psoriasis | 16 | 150 | **11%** | Inflammatory triad |

---

## MM-Skin VQA (5,452 QA pairs)

| Metric | Value |
|--------|-------|
| Containment match | 0.0% (too strict for verbose responses) |
| BERTScore | Pending |

---

## Files

- `fitzpatrick17k/predictions.jsonl` — 1,000 predictions
- `mm_skin_vqa/predictions.jsonl` — 5,452 VQA predictions
- `confusion_triads/predictions.jsonl` — 820 triad predictions

## Pending

- BERTScore for VQA semantic similarity
- Fairness analysis per Fitzpatrick skin type
- Confusion matrix (6x6) for triads
