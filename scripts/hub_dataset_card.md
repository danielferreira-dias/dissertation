---
language:
- en
license: cc-by-nc-sa-4.0
task_categories:
- image-text-to-text
- visual-question-answering
size_categories:
- 10K<n<100K
tags:
- medical
- dermatology
- vision-language
- vlm-finetune
- chain-of-thought
- structured-reasoning
- fairness
pretty_name: Dermatology Reasoning Dataset
configs:
  - config_name: full_reasoning
    data_files:
      - split: train
        path: full_reasoning/train-*
      - split: val
        path: full_reasoning/val-*
  - config_name: label_only
    data_files:
      - split: train
        path: label_only/train-*
      - split: val
        path: label_only/val-*
---

# Dermatology Reasoning Dataset

Structured chain-of-thought training data for fine-tuning vision-language models (VLMs) on dermatological diagnosis. Each row pairs a clinical photograph with an assistant turn that walks through morphology, color, texture, border, location, size, and clinical reasoning before reaching the diagnosis — rather than emitting only a class label.

This repo publishes **only the reasoning annotations** (28,486 rows × 2 formats × 2 splits). The underlying images come from five separate public/research datasets (see *Image sources* below). For redistribution-safe images, see the companion repo [`danielfdias98/derm-reasoning-redistributable`](https://huggingface.co/datasets/danielfdias98/derm-reasoning-redistributable). For the full image set, contact the author for private access.

## Quick load

```python
from datasets import load_dataset

# Default config: full structured reasoning
ds = load_dataset("danielfdias98/derm-reasoning", "full_reasoning")
print(ds)  # DatasetDict {train: 25_637, val: 2_849}

# Label-only ablation
ds_lo = load_dataset("danielfdias98/derm-reasoning", "label_only")
```

## Two training formats (ablation)

| Config | Assistant turn content |
|---|---|
| `full_reasoning` | Diagnosis + category + observation + morphology + color + texture + border + location + size + reasoning + confidence + Fitzpatrick skin type |
| `label_only` | Diagnosis + category only |

The two formats use the same train/val split (95/5, seed=42), so you can directly measure the effect of structured reasoning vs. label-only supervision per architecture.

## Row schema

```python
{
  "messages": [
    {"role": "user", "content": [{"type": "image", "image": "final/train/<class>/<file>", "text": ""},
                                  {"type": "text",  "image": "", "text": "Diagnose this skin condition with structured reasoning."}]},
    {"role": "assistant", "content": [{"type": "text", "image": "", "text": "{\"diagnosis\": \"...\", \"observation\": \"...\", ...}"}]}
  ],
  "image":     "final/train/<class>/<file>",   # relative path used in the messages
  "image_id":  "<basename without extension>",  # for cross-source deduplication
  "class":     "<directory name, e.g. 'psoriasis'>",
  "source":    "scin | pad_ufes | skincap | kaggle_dermnet | dermnet_nz",
}
```

The image path is **relative** — you'll need to fetch the image yourself (see *Getting the images* below).

## Image sources & licensing

This dataset is a union of five public/research dermatology datasets. Per-source attribution is preserved in the `source` column of every row:

| Source | Rows (train) | Original license | Redistribution policy |
|---|---:|---|---|
| **SCIN** (Google Research) | 4,332 | Apache 2.0 | ✅ Redistributable |
| **PAD-UFES-20** (Pacheco et al., 2020) | 1,706 | CC BY 4.0 | ✅ Redistributable with attribution |
| **SkinCAP** (Chou et al., 2024) | 3,607 | CC-BY-NC-SA 4.0 | ⚠ Captions: yes (NC); embedded DDI images: no per Stanford RUA |
| **Kaggle DermNet** (Goel et al.) | 15,221 | "Other" — images sourced from DermNet website | ❌ Original DermNet copyright applies; not redistributed here |
| **DermNet NZ** (scraped) | 771 | DermNet NZ copyright | ❌ Not redistributed here |

The dataset is licensed **CC-BY-NC-SA 4.0** to match the most restrictive component (SkinCAP). Use is for **non-commercial research only**.

### Citations

If you use this dataset, please cite all source datasets:

- **SCIN:** Ward et al. "Crowdsourcing dermatology images with Google Search ads." *Nature Medicine*, 2024.
- **PAD-UFES-20:** Pacheco et al. "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones." *Data in Brief*, 2020.
- **SkinCAP:** Chou et al. "SkinCAP: A Multi-modal Dermatology Dataset Annotated with Rich Medical Captions." *arXiv:2405.18004*, 2024.
- **Kaggle DermNet:** Shubham Goel, "DermNet" (Kaggle), aggregating DermNet atlas images.
- **DermNet NZ:** dermnetnz.org (clinical photo gallery).
- **Fitzpatrick17k** (subset embedded in SkinCAP): Groh et al. "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset." *CVPR Workshop*, 2021.
- **DDI** (subset embedded in SkinCAP, 655 images): Daneshjou et al. "Disparities in dermatology AI performance on a diverse, curated clinical image set." *Science Advances*, 2022.

And cite this dataset itself:

```bibtex
@misc{dias2026derm-reasoning,
  author = {Ferreira Dias, Daniel},
  title  = {Dermatology Reasoning Dataset: Structured chain-of-thought annotations across five public sources},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/datasets/danielfdias98/derm-reasoning}},
}
```

## Getting the images

Three options depending on what's allowed:

### Option 1 — Redistributable subset (~7,500 images, public)

```python
from datasets import load_dataset
imgs = load_dataset("danielfdias98/derm-reasoning-redistributable", split="train")
```

Contains the SCIN + PAD-UFES-20 images only — both publish with redistribution rights.

### Option 2 — Fetch each source independently

For the full ~28k-image set, run the included `download_images.py` script. It walks the `source` column, downloads each source from its original location, and assembles the image folder structure that the JSONL paths expect:

```
final/train/<class>/<file>.{jpg,png}
```

Sources you'll need to register / accept terms for:
- **SCIN** — Google Cloud Storage public bucket (no auth needed)
- **PAD-UFES-20** — Kaggle (`kaggle datasets download mahdavi1202/skin-cancer`)
- **SkinCAP** — HuggingFace (`huggingface-cli download joshuachou/SkinCAP`)
- **Kaggle DermNet** — Kaggle (`kaggle datasets download shubhamgoel27/dermnet`)
- **DermNet NZ** — manual gallery scraping (research-fair-use, please respect their robots.txt)

See the script for details.

### Option 3 — Private full set (request access)

For dissertation defense / examiner access, contact the author at the email in the citation block. Access is granted via a private companion repo (`danielfdias98/derm-reasoning-full`).

## Annotation methodology

The structured reasoning annotations were generated using **Gemini 2.5 Flash** as a label-anchored teacher: the expert ground-truth diagnosis is provided in the prompt, and the model's job is to *describe features that support the known diagnosis*, not to independently diagnose. This prevents label drift and produces consistent, clinically-grounded reasoning. The methodology is described in detail in the dissertation companion paper (in preparation).

Annotations include:
- **`label_match`** — boolean flag for noisy labels (filtered out before publication)
- **`fitzpatrick_skin_type`** — teacher's estimate of FST from the image, providing skin-tone labels for sources that lack them (DermNet, DermNet NZ)
- **Skin-tone-aware language** — the teacher is instructed to adapt color descriptions to the patient's skin tone (e.g. "violaceous" rather than "erythematous" on darker skin), addressing documented bias in dermatological AI

## Splits

```
full_reasoning/train  25,637 rows
full_reasoning/val     2,849 rows
label_only/train      25,637 rows  (same images, abbreviated assistant turn)
label_only/val         2,849 rows
```

The val split is a stratified 5% sample (seed=42) preserving class proportions. There is no separate test split here — the 820-image confusion-triad evaluation set is held out at `final/test/` and not included in this dataset.

## What this dataset is not

- **Not a clinical decision-support tool.** Not for direct deployment in patient care.
- **Not balanced.** Class distribution reflects real-world prevalence; eczema / kaggle-DermNet condition variants dominate.
- **Not equally annotated for fairness.** Per-FST coverage is uneven across source datasets.
