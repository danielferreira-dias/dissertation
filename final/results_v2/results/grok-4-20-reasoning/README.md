# Grok 4.20 Reasoning — Benchmark Results

**Model:** `azure/grok-4-20-reasoning`  
**Deployment:** Azure AI Foundry (`danielfdias98ag-resource`)  
**Date:** 2026-04-11  
**Images processed:** 1,000 (Fitzpatrick17k) + 820 (Confusion Triads)  
**Parse failures:** 0 / 1,820

---

## Fitzpatrick17k (1,000 images)

| Metric | Score |
|--------|------:|
| Top-1 Accuracy | **9.60%** (96/1000) |
| Top-6 Accuracy | **22.90%** (229/1000) |

### Fairness by Fitzpatrick Skin Type (Top-1)

| FST | Correct | Total | Accuracy |
|-----|--------:|------:|---------:|
| FST 1 (lightest) | 15 | 207 | 7.2% |
| FST 2 | 35 | 296 | 11.8% |
| FST 3 | 13 | 197 | 6.6% |
| FST 4 | 23 | 170 | 13.5% |
| FST 5 | 7 | 94 | 7.4% |
| FST 6 (darkest) | 3 | 36 | 8.3% |

---

## Confusion Triads (820 images, 6 classes)

| Metric | Score |
|--------|------:|
| Overall Accuracy | **66.34%** (544/820) |

### Per-Class Accuracy

| Class | Correct | Total | Accuracy |
|-------|--------:|------:|---------:|
| basal_cell_carcinoma | 111 | 150 | 74.0% |
| eczema | 122 | 150 | 81.3% |
| melanoma | 77 | 150 | 51.3% |
| psoriasis | 81 | 150 | 54.0% |
| seborrheic_dermatitis | 50 | 70 | 71.4% |
| seborrheic_keratosis | 103 | 150 | 68.7% |

---

## Comparison with Local Baselines

| Model | Type | Fitz Top-1 | Fitz Top-6 | Triads Acc |
|-------|------|-----------:|-----------:|-----------:|
| medgemma-4b | Local 4B (medical) | 19.70% | 31.10% | 53.29% |
| qwen3.5-9b | Local 9B | 17.30% | 35.00% | **67.07%** |
| qwen3.5-4b | Local 4B | 12.00% | 27.70% | 53.05% |
| **grok-4-20-reasoning** | **API (reasoning)** | **9.60%** | **22.90%** | **66.34%** |
| gemma4-e4b | Local 4B | 5.90% | 22.10% | 45.37% |
| SkinFlow (7B, fine-tuned) | SOTA reference | 29.19% | — | — |

---

## Notes

- **Fitzpatrick17k is an open-ended diagnosis task** (free-form condition name against 335 ground-truth classes). Grok's low Top-1 (9.60%) likely reflects name mismatch and reasoning verbosity rather than poor visual recognition — the model tends to elaborate condition names differently from the ground-truth labels.
- **Confusion Triads is a constrained 6-class task** where Grok performs comparably to Qwen 9B (66.34% vs 67.07%), despite being a reasoning-focused model not specialized for dermatology.
- **Melanoma confusion** (51.3%) is the hardest class — model confuses it with seborrheic keratosis and basal cell carcinoma.
- **Eczema is strongest** (81.3%), consistent with its distinct presentation.
- The model was run at 50 RPM (Azure AI Foundry limit) with `--workers 4 --delay 1`.
