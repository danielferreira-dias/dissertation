# Qwen 3.5 9B — Zero-Shot Baseline Results

- **Model:** `Qwen/Qwen3.5-9B` (9B parameters)
- **Type:** General-purpose natively multimodal VLM (early fusion)
- **Evaluation:** Zero-shot (no fine-tuning)
- **Infrastructure:** NVIDIA L40S (48GB), vLLM 0.19.0, bfloat16, batch size 16
- **Date:** 2026-04-09

---

## Fitzpatrick17k Benchmark (1,000 images, 114 conditions)

| Metric | Value |
|--------|-------|
| **Top-1 accuracy** | **33.0%** |
| **Top-6 accuracy** | **33.0%** (single diagnosis output, no ranked list) |

### Comparison with published baselines

| Model | Parameters | Top-1 | Top-6 | Source |
|-------|-----------|-------|-------|--------|
| **Qwen 3.5 9B (zero-shot)** | **9B** | **33.0%** | **33.0%** | **This work** |
| SkinFlow (fine-tuned) | 7B | 29.19% | 71.16% | Liu et al., 2026 |
| Qwen 3.5 4B (zero-shot) | 4B | 27.9% | 27.9% | This work |
| MedGemma 4B (zero-shot) | 4B | 27.1% | 29.7% | This work |
| Gemma 4 E4B (zero-shot) | ~4B | 19.0% | 20.3% | This work |
| GPT-5.2 | N/A | 18.24% | 42.59% | Liu et al., 2026 |

**Exceeds SkinFlow's SOTA by +3.8% on Top-1 — zero-shot, no fine-tuning, no architecture changes.**

---

## Confusion Triads (820 images, 6 classes)

Pending — benchmark running.

---

## MM-Skin VQA (5,452 QA pairs)

Running — 1,552/5,452 in progress.

---

## Files

- `fitzpatrick17k/predictions.jsonl` — 1,000 predictions

## Pending

- Confusion triads evaluation
- MM-Skin VQA evaluation
- BERTScore for VQA
- Fairness analysis per Fitzpatrick skin type
