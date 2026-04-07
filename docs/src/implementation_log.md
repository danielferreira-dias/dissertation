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

## Phase 3: Training Scope Decision

### 3.1 Scoped vs. Broad Training

An initial approach scoped training to the 6 evaluation classes only (~2,575 images). After researching how published dermatology VLMs handle class scope, we shifted to **broad training across all available conditions**.

**Key insight from literature:** No published dermatology VLM scopes to a fixed class set. SkinGPT-4 (Zhou et al., Nature Communications 2024) trains on 52,929 images across all conditions. LLaVA-Med (Li et al., NeurIPS 2023) uses 600k+ biomedical image-text pairs with open vocabulary. SkinFlow (Liu et al., arXiv 2601.09136, 2026) supports ~200 conditions. The consensus is: **train broadly, evaluate on focused benchmarks.**

**Rationale:** A VLM trained on only 6 classes can only ever output those 6 diagnoses — when shown an unseen condition, it hallucmates one of the 6. Broad training teaches generalizable dermatological reasoning and the ability to express uncertainty on unfamiliar presentations.

### 3.2 Final Split Architecture

| Set | Images | Classes | Purpose |
|-----|--------|---------|---------|
| `final/train/` | 29,913 | 337 | Broad training across all conditions from all datasets |
| `final/test/` | 820 | 6 | Focused evaluation on confusion triads + fairness |

- 820 test images excluded from training (no data leakage)
- Cross-dataset duplicates removed by filename
- Test set preserves the two confusion triads for targeted evaluation

### 3.3 Evaluation Set (6 classes)

| Class | Test Images | Confusion Zone |
|-------|------------|---------------|
| Seborrheic Dermatitis | 70 | Inflammatory triad |
| Psoriasis | 150 | Inflammatory triad |
| Eczema | 150 | Inflammatory triad |
| Seborrheic Keratosis | 150 | Lesion triad |
| Basal Cell Carcinoma | 150 | Lesion triad |
| Melanoma | 150 | Lesion triad |

### 3.4 Class Imbalance in Training

The training set reflects real-world condition prevalence — eczema (2,443) vs. seborrheic dermatitis (236) represents a ~10:1 imbalance. Planned mitigation strategies:
- Standard data augmentation (rotation, flipping, color jitter) at training time
- Class-weighted loss function inversely proportional to class frequency
- Oversampling minority classes with different augmentations per epoch
- Rich structured reasoning text compensates — detailed descriptions carry more training signal than simple labels

---

## Phase 4: Structured Reasoning Pipeline

### 4.1 Motivation from SkinFlow

The prompt design for reasoning generation was directly informed by the SkinFlow ablation study (Liu et al., arXiv 2601.09136, 2026). SkinFlow is a 7B dermatology VLM built on Qwen2.5-VL that achieved +12.06% Top-1 accuracy over GPT-5.2 on the Fitzpatrick17k benchmark. Their ablation revealed:

| Component | Accuracy contribution |
|-----------|----------------------|
| Structured medical captioning (Stage 1 RL) | **+9.23%** |
| Dynamic Vision Encoder architecture change | +4.74% |

**The training strategy contributed nearly twice the improvement of the architecture change.** Specifically, forcing the model to generate structured descriptions with explicit fields (color, location, morphology, size, border, texture) before reasoning about the diagnosis was the dominant factor.

This finding directly shaped our approach: rather than modifying the model architecture (out of scope), we invest in high-quality structured training data — applying the insight that **data quality and structure can compensate for architectural limitations**.

### 4.2 Prompt Design

The reasoning generation prompt uses **label-anchored structured descriptions**, informed by both SkinFlow's structured captioning approach and clinical dermatology examination conventions.

**Key design decisions:**

1. **Label-anchored** — The expert diagnosis is provided to the teacher model. This prevents label drift (the teacher model hallucinating a different diagnosis). The teacher's job is to describe features that support the expert label, not to independently diagnose. This approach follows the principle established in knowledge distillation literature that the teacher should explain the ground truth, not generate new labels (Hinton et al., 2015).

2. **Structured fields** — Inspired by SkinFlow's Stage 1 captioning schema and the ABCDE framework for lesion assessment:
   - `morphology` — primary lesion type and arrangement
   - `color` — adapted to observed skin tone
   - `texture` — surface quality
   - `border` — border characteristics
   - `location` — body location and distribution pattern
   - `size` — estimated lesion size
   - `reasoning` — clinical synthesis tying features to diagnosis

3. **Skin-tone-aware** — The prompt explicitly instructs the teacher to adapt descriptions to the patient's skin tone, avoiding light-skin-centric defaults (e.g., "violaceous" instead of "erythematous" for darker skin). This addresses the documented bias in dermatological AI where models trained on light-skin descriptions fail on FST V-VI presentations (Daneshjou et al., Science Advances 2022).

4. **Label match flag** — A `label_match` boolean allows the teacher to flag images where the visual content clearly contradicts the expert label. This serves as automated quality control for noisy labels, particularly important for the Kaggle DermNet dataset where folder-level labels group multiple conditions.

5. **Fitzpatrick estimation** — The teacher estimates FST from the image, providing skin tone labels for datasets that lack them (DermNet, DermNet NZ).

### 4.3 Structured Output Format

```json
{
  "label_match": true,
  "morphology": "Well-demarcated erythematous plaque with overlying thick silvery-white scale...",
  "color": "Bright erythematous base with silvery-white micaceous scale...",
  "texture": "Thick, adherent, micaceous scale with Auspitz sign potential...",
  "border": "Well-defined, sharply demarcated raised borders...",
  "location": "Extensor surface of the elbow, bilateral...",
  "size": "~5cm plaque...",
  "reasoning": "The well-demarcated plaque with silvery scale on extensor surfaces is characteristic of chronic plaque psoriasis...",
  "confidence": "high",
  "fitzpatrick_skin_type": 2
}
```

This structured format serves dual purposes:
- **Training signal:** Each field teaches the student model a different aspect of clinical examination, matching how dermatologists systematically assess lesions
- **Evaluation metric:** Individual fields can be scored independently, enabling fine-grained analysis of model capabilities (e.g., "the model describes morphology well but struggles with color on dark skin")

### 4.4 Teacher Model Strategy

Multiple teacher models were considered for generating structured descriptions:

| Teacher | Cost | Quality | Reproducible |
|---------|------|---------|-------------|
| Gemini 2.5 Flash | ~$2-5 (or free tier) | Good | No (API) |
| Claude Sonnet 4.6 | ~$15-25 | Better reasoning | No (API) |
| MedGemma 27B (on RunPod) | GPU cost only | Medical-specialized | Yes (fixed weights) |

**Planned approach:** Multi-teacher with open-source primary. MedGemma 27B (Google, trained on dermatology images) as primary teacher for reproducibility, Claude Sonnet as validator for a quality sample, and SkinCAP's existing 4,000 dermatologist-written captions as reference baseline.

**Pre-existing reasoning sources:**
- **SkinCAP** (4,000 images): Dermatologist-written captions available. Not structured in our format but usable as reference for validation.
- **SkinCaRe/SkinCoT** (3,041 images): Clinician-certified chain-of-thought reasoning — access pending. If approved, would serve as gold-standard comparison for teacher model output quality.

### 4.5 Conversion to Training Format

Raw structured descriptions are converted to VLM chat format for Qwen 2.5-VL fine-tuning:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "text", "text": "Analyze this dermatological image..."}
      ]
    },
    {
      "role": "assistant",
      "content": "Diagnosis: Psoriasis\nCategory: Inflammatory\nConfidence: High\nFitzpatrick skin type: 2\nMorphology: Well-demarcated erythematous plaque...\nColor: ...\nTexture: ...\nBorder: ...\nLocation: ...\nSize: ...\nReasoning: ..."
    }
  ]
}
```

The conversion script (`convert_to_training.py`) automatically:
- Injects the expert diagnosis and category from the ground truth label (never from the teacher model)
- Filters out entries where `label_match` is false (noisy labels)
- Formats the structured fields into a consistent assistant response

---

## Phase 5: Evaluation Pipeline

### 5.1 Benchmarks

Three benchmarks were selected to evaluate model performance from different angles:

| Benchmark | Images | Samples | Type | Purpose |
|-----------|--------|---------|------|---------|
| **Fitzpatrick17k 1,000** | 1,000 | 1,000 | Classification (Top-1/Top-6) | Comparability with published work (SkinFlow: 29.19% Top-1, GPT-5.2: 18.24%) |
| **MM-Skin VQA** | 11,002 | 5,452 QA pairs | Open-ended VQA | Tests reasoning quality across 3,781 unique question types |
| **Confusion Triads** | 820 | 820 | 6-class classification | Tests performance on deliberately confusable conditions + fairness per FST |

**Fitzpatrick17k 1,000** was constructed by stratified sampling from the SkinCAP dataset's Fitzpatrick17k subset (3,223 images with valid FST I-VI labels). This is the same benchmark used by SkinFlow (Liu et al., 2026), SkinGPT-4 (Zhou et al., 2024), and other published dermatology VLMs, enabling direct comparison.

**MM-Skin VQA** (Zhao et al., 2025) provides 5,452 question-answer pairs covering diverse question types: "What does this show?", "Where is the lesion?", "How large is it?", "What type of skin condition?". This tests the model's ability to generate clinically relevant free-text responses, not just labels.

**Confusion Triads** is our custom evaluation set designed to test the model's clinical reasoning on deliberately confusable conditions (inflammatory triad + lesion triad).

**Benchmarks considered but not yet available:**
- **DermaBench** (Yilmaz et al., arXiv:2601.14084, 2026): Clinician-annotated VQA benchmark on DDI dataset. Harvard Dataverse DOI reserved but data not yet uploaded.
- **DermEVAL** (Zhao et al., WACV 2026): Dermatologist-reviewed benchmark with 16 categories. No public download found.

### 5.2 Student Models

Four models were selected for evaluation, spanning medical-specialized and general-purpose architectures:

| Model | HuggingFace ID | Size | Type | Rationale |
|-------|---------------|------|------|-----------|
| **MedGemma 1.5 4B IT** | `google/medgemma-1.5-4b-it` | 4B | Medical | Published 62% on dermatology benchmarks. Trained on dermatology images. Strongest medical baseline. |
| **Gemma 4 E4B IT** | `google/gemma-4-E4B-it` | 4B | General | General-purpose baseline from same family as MedGemma. Tests whether medical pre-training matters. |
| **Qwen 3.5 4B** | `Qwen/Qwen3.5-4B` | 4B | General | Small VLM baseline. Tests whether a 4B model can be competitive after fine-tuning. |
| **Qwen 3.5 9B** | `Qwen/Qwen3.5-9B` | 9B | General | Larger VLM. Tests whether scale helps — comparable to SkinFlow's 7B base model. |

### 5.3 Evaluation Methodology

**Phase 1: Zero-shot baseline** — All 4 models are evaluated on all 3 benchmarks without any fine-tuning. This establishes the baseline: how well can each model do dermatology out of the box?

**Phase 2: Post-training evaluation** — After fine-tuning on our 29,913-image training set with structured reasoning, the same models are re-evaluated on the same benchmarks. The improvement (Δ accuracy) measures the effectiveness of our knowledge distillation approach.

**Metrics:**
- **Fitzpatrick17k:** Top-1 accuracy, Top-6 accuracy, accuracy per FST group (fairness)
- **MM-Skin VQA:** Containment match (ground truth appears in model response)
- **Confusion Triads:** Per-class accuracy, confusion matrix between triads, accuracy per FST group

**Fairness analysis:** For Fitzpatrick17k and Confusion Triads, accuracy is broken down by Fitzpatrick skin type (I-VI). A model that performs well on light skin (FST I-II) but poorly on dark skin (FST V-VI) is considered biased, regardless of overall accuracy.

### 5.4 Published LLM Baselines (Fitzpatrick17k)

Since large commercial models cannot be run locally, we use published results from the SkinFlow study (Liu et al., arXiv:2601.09136, 2026) as reference baselines on the Fitzpatrick17k benchmark:

| Model | Parameters | Top-1 Accuracy | Top-6 Accuracy | Source |
|-------|-----------|---------------|---------------|--------|
| GPT-5.2 | N/A (commercial) | 18.24% | 42.88% | SkinFlow (Liu et al., 2026) |
| Qwen3-VL-235B | 235B | 17.13% | 42.59% | SkinFlow (Liu et al., 2026) |
| InternVL3-78B | 78B | — | — | SkinFlow (Liu et al., 2026) |
| SkinFlow (fine-tuned) | 7B | 29.19% | 71.16% | SkinFlow (Liu et al., 2026) |

**Key observation:** GPT-5.2 achieves only 18.24% Top-1 on Fitzpatrick17k despite being orders of magnitude larger than our student models. SkinFlow, a 7B model (same scale as our Qwen 3.5 9B), surpasses it by +10.95% through structured captioning and RL fine-tuning. This establishes the central thesis target:

> If our fine-tuned SLMs (4B-9B) can match or exceed GPT-5.2's 18.24% Top-1 accuracy on Fitzpatrick17k, it demonstrates that knowledge distillation with structured reasoning can compensate for 30x+ parameter reduction — validating the SLM approach for dermatological diagnosis.

The comparison table in the final evaluation will include:
- Published LLM baselines (above)
- Our zero-shot baselines (measured now)
- Our post-training results (measured after fine-tuning)
- Δ improvement per model

### 5.4 Infrastructure

Evaluation requires GPU inference for all 4 models:
- 4B models: ~10 GB VRAM
- 9B model: ~20 GB VRAM
- **RunPod** with A40 (48 GB) or RTX 4090 (24 GB) recommended for consistent evaluation across all models

### 5.5 Pipeline Structure

```
src/eval/
├── config.py          — Models, benchmarks, prompts
├── run_benchmark.py   — Run a model against a benchmark (saves predictions.jsonl)
└── score_results.py   — Score results and generate comparison tables
```

Usage:
```bash
# Run one model on one benchmark
python src/eval/run_benchmark.py --model medgemma-4b --benchmark fitzpatrick17k

# Run all models on all benchmarks
python src/eval/run_benchmark.py --all

# Score and compare results
python src/eval/score_results.py
```

---

## Phase 6: Multi-Agent Architecture Reconciliation

### 6.1 The Problem

The original system design (Chapter 3) proposed 4 specialized SLM+RAG agents, each dedicated to a small group of diseases. The VLM orchestrator would classify the image and route to the appropriate specialist. However, the training scope decision (Phase 3) expanded the VLM to classify 337 conditions across all datasets. This created a mismatch: 337 diseases cannot map to 4 disease-level agents without either creating hundreds of agents (impractical) or discarding the broad training advantage.

### 6.2 The Solution: Category-Level Specialization

Rather than specializing agents by individual disease, the 4 agents specialize by **clinical category** — mirroring how dermatologists reason in practice (category first, then specific diagnosis).

| Agent | Clinical Category | Example Conditions | RAG Knowledge Base |
|-------|------------------|-------------------|-------------------|
| **Inflammatory Agent** | Inflammatory dermatoses | Seb dermatitis, psoriasis, eczema, contact dermatitis, rosacea, urticaria | Literature on inflammatory pathology, scale types, distribution patterns |
| **Malignant Agent** | Malignant/pre-malignant lesions | Melanoma, BCC, SCC, actinic keratosis, Bowen's disease | Literature on ABCDE criteria, dermoscopic features, staging, urgency |
| **Benign Agent** | Benign growths and tumors | Seb keratosis, nevi, cysts, fibromas, dermatofibromas, keloids | Literature on benign lesion characteristics, when to monitor vs remove |
| **Infectious Agent** | Infectious/parasitic conditions | Tinea, scabies, warts, impetigo, herpes, folliculitis | Literature on pathogen identification, distribution patterns, treatment |

### 6.3 Updated Pipeline

```
User Image → VLM Orchestrator (fine-tuned on 337 conditions)
              → outputs: specific diagnosis + clinical category + confidence
              → routes to category-level specialist:

    ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
    │ Inflammatory     │ Malignant       │ Benign          │ Infectious      │
    │ Agent            │ Agent           │ Agent           │ Agent           │
    │ (SLM + RAG)      │ (SLM + RAG)     │ (SLM + RAG)     │ (SLM + RAG)     │
    │                  │                 │                 │                 │
    │ ChromaDB:        │ ChromaDB:       │ ChromaDB:       │ ChromaDB:       │
    │ All inflammatory │ All malignant   │ All benign      │ All infectious  │
    │ conditions       │ conditions      │ conditions      │ conditions      │
    └────────┬────────┴────────┬────────┴────────┬────────┴────────┬────────┘
             └────────────────┴────────────────┴────────────────┘
                                      ↓
                              Validation Agent
                                      ↓
                              Final Report
```

### 6.4 Why This Is Better

1. **Scalable:** 337 diseases map to 4 categories. Adding new diseases only requires updating the RAG knowledge base, not deploying new agents.

2. **Clinically realistic:** Dermatologists reason hierarchically — "Is this inflammatory or neoplastic?" precedes "Is this psoriasis or eczema?" The architecture mirrors this cognitive workflow, as described in dermatological decision-making literature (Habif, Clinical Dermatology, 2020).

3. **RAG handles specificity:** The specialized agent doesn't need to encode all conditions in its weights. It reasons about the category's shared characteristics and retrieves condition-specific context from ChromaDB. This aligns with the knowledge externalization principle established in Chapter 2 — medical knowledge resides in retrievable databases, not model parameters (Gao et al., RAG Survey, 2024).

4. **Confusion triads become a feature:** The Inflammatory Agent must distinguish seb dermatitis vs psoriasis vs eczema — exactly the hard within-category reasoning that a specialized agent with domain-specific RAG should excel at. This validates the multi-agent approach: a generalist would struggle here, but a category specialist with targeted retrieval can leverage subtle distinctions in the medical literature.

5. **Consistent with Chapter 3 structure:** The number of agents (4 specialized + 1 orchestrator + 1 validation = 6) remains identical. The change is in what they specialize on (clinical categories vs individual diseases).

### 6.5 Mapping to Evaluation

The confusion triad evaluation (Phase 5) directly validates this architecture:

- **Inflammatory triad** (seb derm ↔ psoriasis ↔ eczema): All routed to the Inflammatory Agent. Tests whether the VLM correctly identifies the specific condition AND whether the agent provides accurate category-specific reasoning.
- **Lesion triad** (seb keratosis ↔ BCC ↔ melanoma): Routes to Benign Agent or Malignant Agent. Tests whether the VLM correctly distinguishes benign from malignant AND whether routing to the correct agent changes response quality.

The lesion triad is particularly interesting because misrouting (sending a melanoma to the Benign Agent) would produce qualitatively different and potentially dangerous advice — a measurable failure mode that validates the routing architecture.

### 6.6 Implications for Thesis

This architectural decision strengthens the dissertation argument:

> "We demonstrate that a modular multi-agent system with 4 category-specialized agents, each augmented with domain-specific RAG, can handle 337 dermatological conditions while maintaining clinical reasoning quality comparable to monolithic LLM systems. The system scales through knowledge bases rather than model replication — adding coverage for new conditions requires only updating the RAG corpus, not training new agents."

This is a more compelling contribution than a system limited to 4-6 pre-defined diseases, and it directly addresses the scalability criticism that reviewers might raise against a small fixed-class system.

---

## References

- **SkinFlow:** Liu et al. "SkinFlow: Efficient Information Transmission for Open Dermatological Diagnosis via Dynamic Visual Encoding and Staged RL." arXiv:2601.09136, January 2026.
- **SkinGPT-4:** Zhou et al. "Pre-trained multimodal large language model enhances dermatological diagnosis using SkinGPT-4." Nature Communications, 2024.
- **LLaVA-Med:** Li et al. "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day." NeurIPS 2023.
- **SkinCaRe:** "SkinCaRe: A Multimodal Dermatology Dataset Annotated with Medical Caption and Chain-of-Thought Reasoning." arXiv:2405.18004, 2024.
- **SkinCAP:** Chou et al. "SkinCAP: A Multi-modal Dermatology Dataset Annotated with Rich Medical Captions." arXiv:2405.18004, 2024.
- **Knowledge Distillation:** Hinton et al. "Distilling the Knowledge in a Neural Network." NeurIPS Workshop, 2015.
- **Fairness in Dermatology AI:** Daneshjou et al. "Disparities in dermatology AI performance on a diverse, curated clinical image set." Science Advances, 2022.
- **PAD-UFES-20:** Pacheco et al. "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones." Data in Brief, 2020.
- **Fitzpatrick17k:** Groh et al. "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset." CVPR Workshop, 2021.
- **SCIN:** Ward et al. "Crowdsourcing dermatology images with Google Search ads." Nature Medicine, 2024.

---

## Next Steps

1. **Zero-shot baselines:** Run all 4 student models on Fitzpatrick17k + MM-Skin VQA + Confusion Triads (RunPod GPU)
2. **Reasoning generation:** Generate structured descriptions for 29,913 training images using teacher model
3. **Training format conversion:** Convert to VLM chat format
4. **Fine-tuning:** LoRA fine-tune all 4 student models on RunPod GPU
5. **Post-training evaluation:** Re-run all 3 benchmarks, measure Δ improvement + fairness per FST
6. **Teacher comparison:** Compare reasoning quality from different teacher models (MedGemma vs Claude vs Gemini vs clinician ground truth from SkinCoT if approved)
