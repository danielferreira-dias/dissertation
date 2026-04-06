# Implementation Log — Efficient Dermatology VLM Dissertation

This document tracks the practical implementation process for training and evaluating a small vision-language model (VLM) for dermatological skin condition classification with clinical reasoning.

---

## Phase 1: Dataset Collection

### 1.1 Datasets Gathered

Ten public datasets were evaluated. Six were included in the final corpus, four were skipped.

| # | Dataset | Source | Images | Type | Status |
|---|---------|--------|--------|------|--------|
| 1 | **SCIN** | Google Research (GCS) | 5,288 | Smartphone photos | Included |
| 2 | **Kaggle DermNet** | Kaggle (shubhamgoel27/dermnet) | 18,619 | Clinical atlas photos | Included |
| 3 | **PAD-UFES-20** | Mendeley / Kaggle | 2,298 | Smartphone photos | Included |
| 4 | **SkinCAP** | HuggingFace (joshuachou/SkinCAP) | 4,000 | Fitzpatrick17k + DDI | Included |
| 5 | **DermNet NZ** | dermnetnz.org (scraped) | 1,261 | Clinical photos | Included |
| 6 | **Fitzpatrick17k** | GitHub (mattgroh/fitzpatrick17k) | 772 | Clinical atlas | Replaced by SkinCAP |
| 7 | **DermaSynth** | GitHub (abdurrahimyilmaz) | 92,000 | Mixed modalities | Skipped — mostly dermoscopic |
| 8 | **DermEVAL** | WACV 2026 paper | 11,347 | DermNet-sourced | Skipped — evaluation benchmark only |
| 9 | **MM-Skin** | GitHub (ZwQ803) | 11,039 | Textbook + mixed | Skipped — heavy overlap, dermoscopic/pathological |
| 10 | **SkinCaRe/SkinCoT** | HuggingFace (yuhos16) | 3,041 | DermNet + CoT reasoning | Pending access approval |

### 1.2 Dataset Processing Details

**SCIN (Skin Condition Image Network):**
- Downloaded via Google Cloud Storage public bucket
- Labels extracted from `weighted_skin_condition_label` column with confidence threshold >= 0.3
- 22 conditions organized into class folders
- Key value: Fitzpatrick + Monk skin tone labels (both self-reported and dermatologist-assessed)

**Kaggle DermNet:**
- Original dataset groups multiple conditions into 23 broad folders (e.g., "Seborrheic Keratoses and other Benign Tumors" contained keloids, hydrocystomas, cysts)
- Problem: folder-level labels are unreliable for training
- Solution: wrote `organize_dermnet.py` to parse individual **filenames** (e.g., `basal-cell-carcinoma-face-1.jpg`) and extract actual conditions
- Result: classified 18,619 of 19,559 images (95.2%) across 160+ distinct conditions
- 939 unclassified images were niche conditions with non-standard filenames

**PAD-UFES-20:**
- Downloaded from Kaggle mirror (mahdavi1202/skin-cancer)
- Organized using `metadata.csv` diagnostic labels
- Key value: 845 biopsy-proven BCC + 52 biopsy-proven melanoma (gold standard labels)
- Brazilian patient population adds geographic diversity
- Fitzpatrick metadata included (skews FST 1-3)

**SkinCAP:**
- Cloned from HuggingFace via git + git LFS
- 3,345 images from Fitzpatrick17k + 655 from DDI (Diverse Dermatology Images)
- Key value: dermatologist-written English captions for every image
- Also includes 48 binary clinical concept annotations (Plaque, Scale, Erythema, etc.)
- Captions originally in Chinese, translated via GPT-4 — quality varies
- Some label noise detected (e.g., seb dermatitis image described as seb keratosis in caption)

**DermNet NZ (scraped):**
- Scraped using Playwright headless browser from dedicated gallery pages
- 11 conditions: seb dermatitis, psoriasis, atopic dermatitis, acne, BCC, melanoma (4 subtypes), rosacea, vitiligo, scabies, eczema, impetigo
- High-quality clinical close-up photos with DermNet watermarks

**Fitzpatrick17k (replaced):**
- Attempted download via external URLs — 84% of URLs are dead (known issue)
- Only 772 of 4,776 images downloadable
- Replaced by SkinCAP which contains 3,345 Fitzpatrick17k images hosted on HuggingFace (actually downloadable) with captions

### 1.3 Datasets Skipped and Why

- **DermaSynth (92k):** 78% dermoscopic images (different modality from our target), heavy overlap with SCIN/PAD-UFES, Gemini-generated text that we'd replace with our own pipeline anyway
- **DermEVAL (11k):** Evaluation benchmark — using it for training would contaminate evaluation. Bookmarked for model evaluation phase.
- **MM-Skin (11k):** 5 of 6 clinical sources already in our corpus. Unique content is dermoscopic/pathological — different modality.
- **SkinCaRe/SkinCoT (3k):** Clinician-certified chain-of-thought reasoning for DermNet images. Extremely valuable but gated — access request submitted, awaiting author approval.

---

## Phase 2: Class Selection

### 2.1 Selection Philosophy

Rather than selecting maximally distinct classes (which would produce a trivially easy classification task), we deliberately chose classes with **confusable pairs** to test the model's clinical reasoning ability. The goal is a VLM that can reason through diagnostic ambiguity, not just pattern-match.

### 2.2 Confusion Zones

**Inflammatory Triad:**
- Seborrheic Dermatitis ↔ Psoriasis ↔ Eczema
- All present as erythematous, scaly patches
- Differentiation requires reasoning about: scale type (greasy vs silvery), distribution (face/scalp vs extensor vs flexural), morphology details

**Lesion Triad:**
- Seborrheic Keratosis ↔ Basal Cell Carcinoma ↔ Melanoma
- All present as raised or pigmented skin lesions
- Represents three severity levels: benign → slow-growing cancer → potentially fatal
- Differentiation requires reasoning about: ABCDE criteria, "stuck-on" appearance, pearly translucency, border regularity

### 2.3 Final 6 Classes

| Class | Category | Total Available | Confusion Zone |
|-------|----------|----------------|---------------|
| Seborrheic Dermatitis | Inflammatory | 236 | Inflammatory triad |
| Psoriasis | Inflammatory | 1,544 | Inflammatory triad |
| Eczema | Inflammatory | 2,593 | Inflammatory triad |
| Seborrheic Keratosis | Benign | 792 | Lesion triad |
| Basal Cell Carcinoma | Malignant | 1,657 | Lesion triad |
| Melanoma | Malignant | 559 | Lesion triad |

### 2.4 Classes Considered but Dropped

| Class | Images | Why dropped |
|-------|--------|-------------|
| Acne | 678 | Visually distinct from selected classes — no confusion zone overlap |
| Atopic Dermatitis | 556 | Subtype of eczema; redundant with eczema class |
| Contact Dermatitis | 1,139 | Hard to distinguish from eczema without clinical history (not visible in photos) |
| Urticaria | 679 | Transient lesions — wheals appear/disappear in hours, inconsistent in photos |
| Vitiligo | 192 | Too few images; strong fairness class but data insufficient |
| Tinea | 1,390 | Zero captions available; no strong fairness story in literature |
| Rosacea | 320 | Would need acne as confusion pair to be meaningful |

### 2.5 Fairness Relevance

Every selected class has documented skin tone bias:

| Class | Documented bias |
|-------|----------------|
| Seborrheic Dermatitis | Erythema-based diagnosis fails on dark skin (FST V-VI) |
| Psoriasis | Underdiagnosed in dark skin; erythema masked by pigmentation |
| Eczema | Presentation differs significantly across skin tones; follicular prominence on dark skin |
| Seborrheic Keratosis | Dermatosis papulosa nigra variant common in dark skin, often misdiagnosed |
| Basal Cell Carcinoma | Training data >90% light skin; most common cancer globally but underrepresented in dark skin datasets |
| Melanoma | Worst survival disparity across skin tones due to delayed diagnosis in people of color |

---

## Phase 3: Final Dataset Split

### 3.1 Split Configuration

- **Train:** 500 images per class (or all available if < 500)
- **Test:** 150 images per class (or ~30% if total < 650)
- **Random seed:** 42 (reproducible)
- **Deduplication:** by filename across datasets

### 3.2 Final Counts

| Class | Train | Test | Total |
|-------|-------|------|-------|
| Seborrheic Dermatitis | 166 | 70 | 236 |
| Psoriasis | 500 | 150 | 650 |
| Eczema | 500 | 150 | 650 |
| Seborrheic Keratosis | 500 | 150 | 650 |
| Basal Cell Carcinoma | 500 | 150 | 650 |
| Melanoma | 409 | 150 | 559 |
| **Total** | **2,575** | **820** | **3,395** |

Output: `final/train/` and `final/test/`

### 3.3 Class Imbalance

Seborrheic dermatitis (166 train) and melanoma (409 train) are underrepresented. Planned mitigation:
- Standard data augmentation (rotation, flipping, color jitter) at training time
- Class-weighted loss function inversely proportional to class frequency
- Oversampling minority classes with different augmentations per epoch
- Rich reasoning text from distillation compensates — detailed chain-of-thought descriptions carry more training signal than simple labels

---

## Next Steps

1. **Reasoning generation:** Generate clinical reasoning descriptions for all 2,575 training images using label-anchored prompts with a teacher model (Gemini 2.5 Flash, Claude Sonnet, or MedGemma 27B)
2. **Training format conversion:** Convert to VLM chat format (user image+prompt → assistant reasoning)
3. **Fine-tuning:** LoRA fine-tune Qwen 2.5-VL on RunPod GPU
4. **Evaluation:** Test fairness across Fitzpatrick skin types on held-out test set
5. **Teacher comparison:** Compare reasoning quality from different teacher models (open-source medical VLM vs commercial API vs clinician ground truth)
