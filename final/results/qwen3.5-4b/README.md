# Qwen 3.5 4B — Zero-Shot Baseline Results

- **Model:** `Qwen/Qwen3.5-4B` (4B parameters)
- **Type:** General-purpose natively multimodal VLM (early fusion)
- **Evaluation:** Zero-shot (no fine-tuning)
- **Infrastructure:** NVIDIA L40S (48GB), vLLM 0.19.0, bfloat16, batch size 16
- **Date:** 2026-04-09

---

## Fitzpatrick17k Benchmark (1,000 images, 114 conditions)

| Metric | Value |
|--------|-------|
| **Top-1 accuracy** | **27.9%** |

### Comparison with published baselines

| Model | Parameters | Top-1 | Source |
|-------|-----------|-------|--------|
| SkinFlow (fine-tuned) | 7B | 29.19% | Liu et al., 2026 |
| **Qwen 3.5 4B (zero-shot)** | **4B** | **27.9%** | **This work** |
| MedGemma 4B (zero-shot) | 4B | 27.1% | This work |
| Gemma 4 E4B (zero-shot) | ~4B | 19.0% | This work |
| GPT-5.2 | N/A | 18.24% | Liu et al., 2026 |
| Qwen3-VL-235B | 235B | 17.13% | Liu et al., 2026 |

**Only 1.3% below SkinFlow's fine-tuned SOTA — zero-shot.** Exceeds GPT-5.2 by +9.7%.

---

## Confusion Triads (820 images, 6 classes)

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **86.0%** (705/820) |

### Per-class breakdown

| Class | Correct | Total | Accuracy | Confusion Zone |
|-------|---------|-------|----------|---------------|
| Psoriasis | 150 | 150 | **100%** | Inflammatory triad |
| Seborrheic Dermatitis | 70 | 70 | **100%** | Inflammatory triad |
| Eczema | 149 | 150 | **99%** | Inflammatory triad |
| Seborrheic Keratosis | 149 | 150 | **99%** | Lesion triad |
| Basal Cell Carcinoma | 135 | 150 | **90%** | Lesion triad |
| Melanoma | 52 | 150 | **35%** | Lesion triad |

### Key findings

1. **Inflammatory triad essentially solved zero-shot (99-100%).** Qwen 3.5 4B perfectly distinguishes seb dermatitis, psoriasis, and eczema without any fine-tuning — a remarkable result that no other model achieved.

2. **Lesion triad strong except melanoma (90-99% vs 35%).** BCC and seb keratosis are well-recognized, but melanoma remains the critical weak point across all models tested.

3. **Melanoma at 35% is the best so far** but still clinically inadequate. Fine-tuning must specifically target melanoma recognition with structured ABCDE reasoning.

4. **Qwen 3.5's native multimodal architecture (early fusion)** appears to provide a fundamental advantage for visual medical tasks compared to adapter-based architectures (Gemma, MedGemma).

---

## MM-Skin VQA (5,452 QA pairs)

| Metric | Value |
|--------|-------|
| Predictions completed | 5,452/5,452 |
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
