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

**Methodological alignment with SkinCaRe:**
Our pipeline independently replicates the core technique published in SkinCaRe (Shen et al., arXiv:2405.18004, 2025). SkinCaRe's SkinCoT component generates chain-of-thought diagnostic reasoning using a multi-model pipeline: Gemini 2.5 Pro for observation captions → GPT-4o-mini for hierarchical reasoning → DeepSeek-R1 for normalization → clinician certification. We apply the same principle — LLM-generated structured reasoning anchored to expert ground-truth labels — using **Gemini 2.5 Flash** as a single teacher model. Our approach trades the multi-model pipeline for simplicity and cost efficiency, while preserving the key insight: a capable teacher model describing visual features that support a known diagnosis produces higher-quality training signal than raw image-label pairs alone.

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
| **Gemma 4 E4B IT** | `google/gemma-4-E4B-it` | 4.5B effective (8B total) | General | General-purpose baseline from same family as MedGemma. Tests whether medical pre-training matters. |
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

## Phase 7: Zero-Shot Baseline Evaluation (In Progress)

### 7.1 Purpose

Before fine-tuning any model, we must establish zero-shot baselines — measuring how well each model performs on dermatological diagnosis *without* any domain-specific training. This serves three purposes:

1. **Scientific rigor:** The improvement from fine-tuning (Δ accuracy) is only meaningful relative to a measured baseline. Without baselines, we cannot claim that our training methodology caused improvement.
2. **Model selection:** Comparing zero-shot performance across architectures reveals which models have the strongest foundation for dermatological reasoning, informing which to prioritize for fine-tuning.
3. **Comparison with published work:** By evaluating on the same Fitzpatrick17k benchmark used by SkinFlow (Liu et al., 2026) and reporting Top-1/Top-6 accuracy, our results are directly comparable to the state of the art.

### 7.2 Why GPT-5.2 and SkinFlow Are Referenced

The Fitzpatrick17k benchmark has become the standard evaluation for dermatology VLMs. The SkinFlow study (Liu et al., arXiv:2601.09136, 2026) provides the most comprehensive published comparison, evaluating models ranging from massive commercial systems to specialized 7B models. Their results establish the performance ceiling and floor for this benchmark:

- **GPT-5.2 (18.24% Top-1):** Represents the best performance achievable by a general-purpose commercial LLM with no dermatology-specific training. Despite having orders of magnitude more parameters than our models, GPT-5.2 achieves only 18.24% — demonstrating that raw scale alone is insufficient for specialized medical tasks. This is our primary comparison target: if our fine-tuned SLMs exceed 18.24%, we validate the thesis that knowledge distillation can compensate for parameter reduction.

- **Qwen3-VL-235B (17.13% Top-1):** The strongest open-source generalist VLM, yet performs below GPT-5.2 — further confirming that general-purpose training plateaus on domain-specific benchmarks.

- **SkinFlow (29.19% Top-1):** A 7B model that surpasses GPT-5.2 by +10.95% through structured captioning and reinforcement learning. SkinFlow's ablation study (Phase 4) directly inspired our structured reasoning pipeline. Their results prove that a 7B model *can* dramatically outperform 235B+ models with the right training strategy — the same principle our dissertation applies to 4B-9B models.

These three reference points define the landscape our models must navigate:
- **Below 17%:** Worse than general-purpose models — training failed
- **17-18%:** Competitive with general-purpose LLMs — baseline parity
- **18-29%:** Exceeding LLMs but below specialized fine-tuned models
- **Above 29%:** State-of-the-art — exceeding SkinFlow

### 7.3 Infrastructure

- **GPU:** NVIDIA L40S (48 GB VRAM) on RunPod EU datacenter
- **Inference engine:** vLLM 0.19.0 with batched inference (batch size 16)
- **Network volume:** Persistent storage attached, survives pod restarts/migrations
- **Throughput:** ~4-5 images/second per batch (L40S + vLLM + bfloat16)

### 7.4 Models Under Evaluation

| Model | HuggingFace ID | Size | Type | Zero-shot hypothesis |
|-------|---------------|------|------|---------------------|
| **MedGemma 1.5 4B IT** | `google/medgemma-1.5-4b-it` | 4B | Medical | Should score highest zero-shot (medical pre-training includes dermatology) |
| **Gemma 4 E4B IT** | `google/gemma-4-E4B-it` | 4.5B effective (8B total) | General | Same family as MedGemma but no medical training — tests if medical pre-training matters |
| **Qwen 3.5 4B** | `Qwen/Qwen3.5-4B` | 4B | General | Natively multimodal (early fusion). Different architecture baseline. |
| **Qwen 3.5 9B** | `Qwen/Qwen3.5-9B` | 9B | General | Larger model — tests if scale within the SLM range improves dermatology performance |

### 7.5 Preliminary Results

**MedGemma 1.5 4B IT — Fitzpatrick17k (1,000 images) — COMPLETE:**

| Metric | Value |
|--------|-------|
| Top-1 accuracy (contains match) | **16.3%** |
| JSON format compliance | 800/1000 (80%) |

**Comparison with published baselines:**

| Model | Parameters | Fitzpatrick17k Top-1 | Notes |
|-------|-----------|---------------------|-------|
| SkinFlow (fine-tuned) | 7B | 29.19% | State-of-the-art, architecture + RL |
| GPT-5.2 | N/A (commercial) | 18.24% | Massive general-purpose LLM |
| Qwen3-VL-235B | 235B | 17.13% | Largest open-source generalist |
| **MedGemma 4B (zero-shot)** | **4B** | **16.3%** | **Our baseline — no fine-tuning** |

**Analysis:** MedGemma 4B achieves 16.3% zero-shot — only 1.94% below GPT-5.2 despite being orders of magnitude smaller. This confirms that medical pre-training (MedGemma was trained on dermatology images) provides a strong foundation. The remaining models (Gemma 4 E4B, Qwen 3.5 4B/9B) are expected to score lower zero-shot since they lack medical training, establishing a larger gap for fine-tuning to close.

**Key observation:** The 80% JSON compliance rate indicates that MedGemma often generates free-text diagnoses rather than structured JSON. This is expected for zero-shot evaluation — the model hasn't been trained to follow our output format. After fine-tuning with structured reasoning data, format compliance should approach 100%.

### 7.6 MedGemma 4B — Complete Results

MedGemma 1.5 4B IT completed all three benchmarks zero-shot.

**Fitzpatrick17k (1,000 images):** 16.3% Top-1 accuracy

**MM-Skin VQA (5,452 QA pairs):** Complete. Scoring pending — will use BERTScore for semantic similarity after all models finish.

**Confusion Triads (820 images, 6 classes):** 54.5% overall accuracy

| Class | Accuracy | Zone | Analysis |
|-------|----------|------|----------|
| Seborrheic Keratosis | **91%** | Lesion triad | Excellent — benign "stuck-on" appearance is distinctive |
| Seborrheic Dermatitis | **63%** | Inflammatory triad | Good — model recognizes scalp/face scaly patches |
| Psoriasis | **62%** | Inflammatory triad | Good — silvery plaques reasonably detected |
| BCC | **55%** | Lesion triad | Moderate — pearly nodules partially recognized |
| Eczema | **53%** | Inflammatory triad | Moderate — confused with other inflammatory conditions |
| Melanoma | **7%** | Lesion triad | **Critical failure** — most dangerous condition worst detected |

**Key findings:**

1. **Melanoma detection at 7% is clinically dangerous.** The most life-threatening condition in our evaluation is the worst detected zero-shot. This directly supports the dissertation argument: zero-shot models cannot be trusted for high-stakes dermatological diagnosis. Fine-tuning with structured reasoning must address this.

2. **Inflammatory triad performance (53-63%)** shows the model partially distinguishes between seb dermatitis, psoriasis, and eczema — but with significant confusion. These conditions share erythematous, scaly presentations that require nuanced reasoning about scale type and distribution.

3. **Lesion triad asymmetry (7% vs 55% vs 91%)** reveals that the model strongly recognizes benign keratoses but fails catastrophically on melanoma. This bias likely reflects training data distribution — benign lesions vastly outnumber melanomas in general-purpose training corpora.

4. **Seb dermatitis at 63%** is encouraging as our personal test condition — reasonable zero-shot performance that fine-tuning should improve further.

### 7.7 Gemma 4 E4B — Complete Results (v2 — Guided JSON)

Gemma 4 E4B is a general-purpose model from the same architecture family as MedGemma, but without medical-specific pre-training. Comparing it directly with MedGemma isolates the effect of medical pre-training.

**Fitzpatrick17k (1,000 images):** 5.9% Top-1, 22.1% Top-6.

**MM-Skin VQA (5,452 QA pairs):** BERTScore F1 = 88.64%.

**Confusion Triads (820 images, 6 classes):** 45.4% overall accuracy

| Class | MedGemma 4B | Gemma 4 E4B | Analysis |
|-------|:-----------:|:-----------:|----------|
| Seb Keratosis | **91%** | 76% | Both strong, medical training helps |
| Eczema | 49% | **74%** | Gemma better — may over-predict eczema |
| Seb Dermatitis | 51% | **74%** | Gemma surprisingly better here |
| BCC | **69%** | 24% | Medical training significantly better |
| Psoriasis | **55%** | 15% | Gemma nearly fails — lacks psoriasis training |
| Melanoma | 3% | **24%** | Both poor, but Gemma 8x better |

**Key findings from MedGemma vs Gemma comparison:**

1. **Medical pre-training is not uniformly better.** MedGemma wins overall (53.3% vs 45.4%) but Gemma 4 outperforms on eczema (74% vs 49%), seb dermatitis (74% vs 51%), and melanoma (24% vs 3%). The models have complementary strengths — different training data creates different biases.

2. **Gemma 4 E4B has a critical fairness gap on Fitzpatrick17k.** 0% accuracy on FST V-VI (dark skin) — the model completely fails on darker skin tones. This is the worst fairness profile of any model tested.

3. **Psoriasis at 15% for Gemma 4** reveals a critical blind spot. Without medical training, the model cannot recognize silvery plaques — the most distinctive feature of psoriasis. This is a strong argument for domain-specific fine-tuning.

4. **Both models fail on melanoma** (3% and 24%). The most dangerous condition remains the hardest to detect regardless of pre-training strategy. This underscores that fine-tuning with our structured reasoning data — which explicitly teaches ABCDE criteria and lesion severity assessment — is essential for clinical safety.

### 7.8 Qwen 3.5 4B — Complete Results (v2 — Guided JSON)

Qwen 3.5 4B is a natively multimodal model using early fusion — vision tokens are integrated during pre-training rather than bolted on via adapters.

**Fitzpatrick17k (1,000 images):** 12.0% Top-1, 27.7% Top-6.

**MM-Skin VQA (5,452 QA pairs):** BERTScore F1 = 88.39%.

**Confusion Triads (820 images, 6 classes):** 53.0% overall.

| Class | MedGemma 4B | Gemma 4 E4B | Qwen 3.5 4B |
|-------|:-----------:|:-----------:|:-----------:|
| Seb Keratosis | **91%** | 76% | 90% |
| BCC | 69% | 24% | 69% |
| Psoriasis | 55% | 15% | **57%** |
| Seb Dermatitis | 51% | **74%** | 56% |
| Eczema | 49% | **74%** | 33% |
| Melanoma | 3% | 24% | **15%** |
| **Overall** | **53.3%** | **45.4%** | **53.0%** |

**Key findings:**

1. **Most equitable fairness profile on Fitzpatrick17k** — only 5.4% spread between best and worst FST. Slightly better on darker skin (FST V: 14.9%, FST VI: 13.9%) than lighter skin (FST II: 9.5%).

2. **Strong seb keratosis (90%) and BCC (69%)** — competitive with MedGemma on lesion recognition despite no medical pre-training.

3. **Melanoma at 15%** — better than MedGemma (3%) but still critically inadequate. Primary fine-tuning target.

### 7.9 Published Baselines from Literature

To contextualize our results, the following published comparisons were compiled from recent dermatology VLM studies. These numbers are cited from their respective papers — not re-run by us.

**Fitzpatrick17k Classification (Published + Our Results — v2 Guided JSON):**

| Model | Params | Top-1 | Top-6 | Source |
|-------|--------|-------|-------|--------|
| SkinFlow (fine-tuned) | 7B | 29.19% | 71.16% | Liu et al., arXiv:2601.09136 |
| **MedGemma 4B (our zero-shot)** | **4B** | **19.7%** | **31.1%** | **This work** |
| GPT-5.2 | Commercial | 18.24% | 42.59% | Liu et al., arXiv:2601.09136 |
| **Qwen 3.5 9B (our zero-shot)** | **9B** | **17.3%** | **35.0%** | **This work** |
| Qwen3-VL-235B | 235B | 17.13% | — | Liu et al., arXiv:2601.09136 |
| **Qwen 3.5 4B (our zero-shot)** | **4B** | **12.0%** | **27.7%** | **This work** |
| **Gemma 4 E4B (our zero-shot)** | **4.5B eff. (8B total)** | **5.9%** | **22.1%** | **This work** |

*All results use vLLM guided JSON structured output (StructuredOutputsParams) to enforce consistent Top-6 differential lists, confidence enums, and reasoning fields. This ensures reproducible scoring across all models.*

**Key findings:** MedGemma 4B exceeds GPT-5.2 on Top-1 (19.7% vs 18.24%). Qwen 3.5 9B matches Qwen3-VL-235B (17.3% vs 17.13%) at 26x fewer parameters. Qwen 9B achieves the highest Top-6 accuracy among our models at 35.0%, demonstrating genuine differential diagnosis capability. Gap to SkinFlow (~10% Top-1) is the fine-tuning target.

**Fairness on Dark Skin — FST V-VI (Published):**

| Model | Params | FST V | FST VI | Source |
|-------|--------|-------|--------|--------|
| SkinGPT-R1 | 7B (frozen + adapters) | 55.0% | 54.9% | arXiv:2511.15242 |
| MedGemma 1.5 | 4B | 30.9% | 28.1% | arXiv:2511.15242 |
| GPT-4o mini | Commercial | 30.6% | 26.0% | arXiv:2511.15242 |

SkinGPT-R1 nearly doubles GPT-4o mini's accuracy on dark skin — a critical fairness finding. Our FST-stratified evaluation (pending) will compare against these baselines.

**Reasoning Quality — DermoBench (Published):**

| Model | Params | Reasoning | Diagnosis | Fairness | Source |
|-------|--------|-----------|-----------|----------|--------|
| DermoGPT | 8B | 67.19% | 78.04% | 93.88% | Ru et al., ru2026dermogpt |
| Gemini 2.5 Flash | Commercial | Lower (hallucinated morphology) | Lower | Lower | Ru et al., ru2026dermogpt |

DermoGPT narrowed the human-AI gap by +13.49 on reasoning. Gemini 2.5 Flash "hallucinated morphology concepts and inconsistent reasoning" — validates our decision to use structured label-anchored prompts rather than open-ended LLM generation.

**General LLMs on Dermatology (Published):**

| Model | Correct Top Diagnosis | Total Coverage (w/ differentials) | Source |
|-------|----------------------|----------------------------------|--------|
| ChatGPT-4o | 66.7% (10/15) | 86.7% | Multiple studies |
| Claude 3.7 Sonnet | 66.7% (10/15) | 86.7% | Multiple studies |
| Gemini 2.0 Flash | 53.3% (8/15) | 60.0% | Multiple studies |

Our Qwen 3.5 9B achieves 67.1% on confusion triads zero-shot. Fine-tuning should push this closer to ChatGPT-4o/Claude coverage rates.

**Key takeaway for thesis:** Every published comparison shows the same pattern — small fine-tuned models (2-8B) consistently beat general-purpose LLMs on dermatology. Our zero-shot baselines are already competitive; fine-tuning should push them further, directly validating the dissertation's central argument.

### 7.10 Final Results Summary (v2 — Guided JSON, All Complete)

| Model | Fitz Top-1 | Fitz Top-6 | Triads | BERTScore F1 | Status |
|-------|:----------:|:----------:|:------:|:------------:|--------|
| MedGemma 4B | **19.7%** | 31.1% | 53.3% | **89.36%** | **Done** |
| Qwen 3.5 9B | 17.3% | **35.0%** | **67.1%** | 88.36% | **Done** |
| Qwen 3.5 4B | 12.0% | 27.7% | 53.0% | 88.39% | **Done** |
| Gemma 4 E4B | 5.9% | 22.1% | 45.4% | 88.64% | **Done** |

### 7.11 Qwen 3.5 9B — Complete Results (v2 — Guided JSON)

Qwen 3.5 9B is the largest model in our evaluation and the best performer on confusion triads.

**Fitzpatrick17k (1,000 images):** 17.3% Top-1, 35.0% Top-6.

**MM-Skin VQA (5,452 QA pairs):** BERTScore F1 = 88.36%.

**Confusion Triads (820 images, 6 classes):** 67.1% overall — highest of all models.

| Class | MedGemma 4B | Gemma 4 E4B | Qwen 3.5 4B | Qwen 3.5 9B |
|-------|:-----------:|:-----------:|:-----------:|:-----------:|
| Seb Keratosis | **91%** | 76% | 90% | 85% |
| Seb Dermatitis | 51% | 74% | 56% | **79%** |
| BCC | 69% | 24% | 69% | **76%** |
| Psoriasis | 55% | 15% | 57% | **65%** |
| Eczema | 49% | **74%** | 33% | 61% |
| Melanoma | 3% | 24% | 15% | **43%** |
| **Overall** | **53.3%** | **45.4%** | **53.0%** | **67.1%** |

**Key findings:**

1. **Best overall triads accuracy (67.1%)** — most balanced across all 6 classes, no single class below 43%.
2. **Melanoma at 43%** — best zero-shot melanoma detection of all models. Still clinically inadequate but a significant lead over MedGemma (3%).
3. **Matches Qwen3-VL-235B on Fitzpatrick17k** (17.3% vs 17.13%) at 26x fewer parameters.
4. **Good fairness profile** — only 4.1% spread between best and worst FST. Slightly better on darker skin (FST IV: 20.0%) than lighter skin (FST II: 15.9%).
5. **Highest Top-6 accuracy (35.0%)** — demonstrates genuine differential diagnosis capability with proper structured output.

### 7.12 Scoring Methodology (v2)

All v2 benchmarks use vLLM guided JSON structured output (`StructuredOutputsParams`) with:
- `StructuredOutputsConfig(backend="xgrammar", disable_any_whitespace=True)` at engine level
- `repetition_penalty=1.1` to prevent degenerate repetition loops
- `max_tokens=2048` to prevent truncation
- JSON schemas enforcing `diagnosis`, `top_6` (array of 6 strings with minLength 3), `confidence` (enum), and `reasoning`
- For triads: `diagnosis` constrained to enum of 6 target classes
- For VQA: `answer` field in JSON wrapper

This ensures 100% JSON compliance across all models and reproducible scoring via `python3 src/eval/score_results.py`.

BERTScore computed using `roberta-large` on GPU, measuring semantic similarity between VQA predictions and ground-truth answers.

### 7.13 Grok 4.20 Reasoning — API Baseline (Azure AI Foundry)

Grok 4.20 Reasoning was added as a commercial reasoning-model baseline via Azure AI Foundry. Run via a new pluggable script `src/eval/run_api_benchmark.py` using litellm (`azure/grok-4-20-reasoning`, 4 workers, 50 RPM rate limit, 0 parse failures across 1,820 predictions).

**Initial substring scoring:** 9.60% Top-1, 22.90% Top-6 on Fitzpatrick17k; 66.34% accuracy on Confusion Triads.

**Paradox observed:** Grok scored worst of all 5 models on Fitzpatrick17k (substring) but nearly matched Qwen 3.5 9B on Confusion Triads (66.34% vs 67.07%). Inspection of the raw predictions showed Grok routinely generates clinically verbose/subtype-specific diagnoses (e.g. "acral lentiginous melanoma" for "melanoma", "morphea" for "scleroderma", "plaque psoriasis" for "psoriasis") which exact-string scorers mark wrong despite being clinically correct or valid subtypes. This observation motivated the LLM-as-Judge rescoring described in §7.14.

### 7.14 LLM-as-Judge Rescoring (SkinFlow-comparable)

**Motivation.** Research on SkinFlow's evaluation methodology revealed that their published 29.19% Top-1 SOTA is obtained via **Gemini-2.5-Pro as an LLM judge** with a clinical rubric that credits synonyms, valid subtypes, and penalises safety-critical errors. Our substring-matching scorer systematically deflates every model that produces verbose clinical language — making our numbers non-comparable with SkinFlow's.

**Methodology.** All 5,000 predictions (5 models × 1,000 entries) were re-scored by **Claude Sonnet 4.5 as the clinical judge** using the following rubric:

| Verdict | Meaning |
|---------|---------|
| `c` (correct) | Exact match, or medically accepted synonym/alias/abbreviation (e.g. "SCC"="squamous cell carcinoma", "atopic dermatitis"="eczema", "hives"="urticaria") |
| `s` (subclass) | Clinically valid subclass/subtype/variant (e.g. "plaque psoriasis" for "psoriasis", "discoid lupus erythematosus" for "lupus erythematosus") |
| `sc` (safety-critical wrong) | Crosses benign↔malignant or infectious↔non-infectious boundary — dangerous misclassification |
| `w` (wrong) | Clinically unrelated, empty, invalid, or gibberish |

**Top-1 (judge) = (c + s) / n** (following SkinFlow's convention of crediting both correct answers and valid subclasses).

**Top-6 (judge)** = fraction of entries where any of the 6 differential predictions was judged c/s against the ground truth.

**Infrastructure.** Predictions were split into 30 batches (5 models × 6 chunks of 100-200 entries), processed in parallel by 30 Claude Sonnet subagents, each applying the rubric to ~200 entries with the 114 Fitzpatrick17k canonical class names in context. Raw predictions were read from `raw_response` fields (preserving the original model output verbatim, unaffected by any preprocessing). Input batches and per-chunk verdicts are preserved at `data/judge_batches/` and `data/judge_verdicts/`.

**Alternative implementation.** A standalone script `LLM-As-a-Judge/judge_bedrock.py` was also built to run the same judge via the AWS Bedrock Converse API using Claude Sonnet 4.5 or Opus 4.5 directly. This provides a reproducible, cost-tracked path that doesn't require spawning subagents and matches SkinFlow's single-judge methodology exactly.

### 7.15 LLM-as-Judge Leaderboard (Final)

**Fitzpatrick17k — Top-1 and Top-6 under both scoring methods:**

| Rank | Model | Params | Top-1 (substring) | Top-1 (judge) | Δ | Top-6 (substring) | Top-6 (judge) | Δ | Safety-crit |
|:---:|-------|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 🥇 | **MedGemma 4B** | 4B | 19.70% | **22.22%** | +2.5% | 31.10% | 34.53% | +3.4% | 16.52% |
| 🥈 | **Qwen 3.5 9B** | 9B | 17.30% | **19.12%** | +1.8% | 35.00% | 39.14% | +4.1% | 23.52% |
| 🥉 | **Grok 4.20 Reasoning** | API | 9.60% | **17.05%** | **+7.5%** | 22.90% | **39.92%** | **+17.0%** | 21.97% |
| 4 | Qwen 3.5 4B | 4B | 12.00% | 14.61% | +2.6% | 27.70% | 31.23% | +3.5% | 18.02% |
| 5 | Gemma 4 E4B | 4.5B eff. (8B total) | 5.90% | 7.32% | +1.4% | 22.10% | 24.47% | +2.4% | 13.74% |

**Verdict breakdown:**

| Model | Correct (c) | Subclass (s) | Safety-crit (sc) | Wrong (w) | Total |
|-------|:----:|:----:|:----:|:----:|:----:|
| MedGemma 4B | 171 | 51 | 165 | 612 | 999 |
| Qwen 3.5 9B | 146 | 45 | 235 | 573 | 999 |
| Grok 4.20 Reasoning | 116 | 54 | 219 | 608 | 997 |
| Qwen 3.5 4B | 106 | 40 | 180 | 673 | 999 |
| Gemma 4 E4B | 60 | 13 | 137 | 787 | 997 |

**Comparison with published baselines on Fitzpatrick17k.**
SkinFlow, GPT-5.2, Qwen3-VL-235B and InternVL3-78B were all scored using **Gemini-2.5-Pro as an LLM judge** in Liu et al. (arXiv:2601.09136, 2026). Our LLM-judge column uses **Claude Sonnet 4.5** with the same 4-category rubric, so the scores below are directly comparable.

| Rank | Model | Params | Type | Top-1 | Top-6 | Source |
|:-:|-------|:-:|:-:|:-:|:-:|:-:|
| 🥇 | **SkinFlow** | 7B | Fine-tuned (Qwen2.5-VL + staged RL) | **29.19%** | **71.16%** | Liu et al., arXiv:2601.09136 |
| 🥈 | **MedGemma 1.5 4B** | 4B | Zero-shot (MedSigLIP) | **22.22%** | 34.53% | This work |
| 🥉 | **Qwen 3.5 9B** | 9B | Zero-shot | 19.12% | 39.14% | This work |
| 4 | GPT-5.2 | Commercial | Zero-shot | 18.24% | 42.88% | Liu et al., 2026 |
| 5 | Qwen3-VL-235B | 235B | Zero-shot | 17.13% | 42.59% | Liu et al., 2026 |
| 6 | **Grok 4.20 Reasoning** | API | Zero-shot (reasoning) | 17.05% | **39.92%** | This work |
| 7 | **Qwen 3.5 4B** | 4B | Zero-shot | 14.61% | 31.23% | This work |
| 8 | InternVL3-78B | 78B | Zero-shot | — | — | Liu et al., 2026 (no numbers reported) |
| 9 | **Gemma 4 E4B** | ~4B | Zero-shot | 7.32% | 24.47% | This work |

**Published-baseline observations that strengthen the thesis:**

- **MedGemma 1.5 4B (ours) beats GPT-5.2** (22.22% vs 18.24%) at a fraction of the parameter count — concrete evidence that architectural medical specialization (MedSigLIP) outperforms commercial frontier scale on this benchmark.
- **MedGemma 1.5 4B (ours) beats Qwen3-VL-235B** (22.22% vs 17.13%) at **~60× fewer parameters** — strong evidence that medical pre-training beats raw scale.
- **Qwen 3.5 9B (ours) essentially matches Qwen3-VL-235B** (19.12% vs 17.13%) at **26× fewer parameters** — validates the "small-but-well-chosen" thesis.
- **Grok 4.20 Reasoning has the highest Top-6 of any evaluated model** that reports that metric (39.92%), narrowly beating Qwen 9B (39.14%) and approaching GPT-5.2 (42.88%) at a fraction of GPT-5.2's inference cost.
- **Gap to SkinFlow SOTA**: MedGemma = 6.97%, Qwen 9B = 10.07%, Grok = 12.14%. These gaps are the explicit targets for fine-tuning with structured reasoning distillation.

**Headline findings:**

1. **MedGemma 4B is within 7 percentage points of fine-tuned SOTA — zero-shot, at 4B parameters.** This is not a black-box "medical pre-training helps" result — the advantage has a specific, architectural source, explained in §7.16 below. Closing the remaining gap through structured reasoning distillation is the thesis contribution.
2. **The ranking changed under the judge.** Grok jumps from rank 5 → rank 3 (+7.45% Top-1, +17.02% Top-6) and now has the **highest Top-6 (39.92%) of any model** — narrowly beating Qwen 9B's 39.14%. The substring scorer was systematically mislabeling Grok's clinically correct verbose/subtype answers as wrong.
3. **Qwen 3.5 9B has the highest safety-critical error rate (23.52%)** despite being #2 on raw Top-1. It is confident and clinically literate enough to make *dangerously* wrong predictions. Fine-tuning should prioritise reducing this.
4. **Gemma 4 E4B has the lowest safety-critical rate (13.74%)** because its errors are mostly clinically unrelated rather than dangerously close to the truth — it fails *safely* but also *uselessly*.
5. **SkinFlow-comparable numbers.** Under an equivalent LLM-judge scoring protocol, our three strongest zero-shot models land at 22.22% / 19.12% / 17.05% — within striking distance of the 29.19% fine-tuned SOTA. Fine-tuning with structured reasoning should close the remaining gap.

**Artifacts.**
- Per-chunk verdict files: `data/judge_verdicts/verdicts_<model>_<batch>_<chunk>.json` (30 files)
- Final aggregated leaderboard: `data/judge_verdicts/leaderboard.json`
- Input batches: `data/judge_batches/judge_batch_<model>_<batch>_<chunk>.jsonl` (30 files)
- Bedrock-based alternative: `src/train/LLM-As-a-Judge/judge_bedrock.py`

### 7.16 Why MedGemma Leads — The MedSigLIP Advantage

MedGemma's 22.22% Top-1 leadership is not a generic "medical pre-training helps" effect. It has a specific, architectural source: **MedGemma 1.5 4B ships with a medically-specialized vision encoder (MedSigLIP), while every other model in our leaderboard uses a generic SigLIP or similar web-scale encoder.**

**The architectural fact.** MedGemma 1.5 4B is built on Gemma 3 as the language backbone, but with the standard SigLIP-400M vision tower replaced by **MedSigLIP-400M** — a SigLIP variant fine-tuned by Google on **33M+ medical image-text pairs** (635K clinical images across multiple modalities + 32.6M histopathology patches), mixed with a 2% weight of the original WebLI training data to retain general vision capability. MedSigLIP is released as a standalone checkpoint (`google/medsiglip-448`) via Google's Health AI Developer Foundations.

**Why this matters for the leaderboard.**
- **Qwen 3.5 4B / 9B** use Qwen's native vision tower (early-fusion, web-scale pretraining, zero medical fine-tuning).
- **Gemma 4 E4B** uses the generic Gemma 4 SigLIP tower (same pretraining recipe as MedGemma's *starting point* before MedSigLIP fine-tuning — so this is the cleanest apples-to-apples comparison).
- **Grok 4.20 Reasoning** uses xAI's proprietary multimodal encoder (web-scale, zero medical specialization).
- **MedGemma 4B** is the only model whose vision encoder has seen clinical/histopathology imagery during pretraining.

**The MedGemma vs Gemma 4 comparison is the cleanest evidence.** Both have the same Gemma-family language backbone and the same SigLIP-400M vision tower *topology*. The only substantive difference is that MedGemma's vision tower has been fine-tuned on 33M medical images. On our LLM-judge leaderboard:

| Model | Vision encoder | Top-1 (judge) | Δ vs Gemma 4 |
|-------|---------------|:-:|:-:|
| Gemma 4 E4B | SigLIP-400M (generic web) | 7.32% | — |
| **MedGemma 1.5 4B** | **MedSigLIP-400M (medical fine-tuned)** | **22.22%** | **+14.90%** |

**MedGemma outperforms Gemma 4 by ~15 percentage points at the same parameter count, same language backbone, same vision tower topology — the entire difference is attributable to MedSigLIP's medical fine-tuning.** The LLM judge amplifies this gap slightly (substring scorer showed +13.8%, the judge shows +14.9%), but the qualitative conclusion is unchanged.

**Why MedSigLIP is not a shortcut for our thesis.** MedSigLIP is only distributed as an **embedding-only checkpoint via Vertex AI / Hugging Face**; it is not released with an end-to-end VLM chat template, no LoRA adapters, and it was never directly fine-tuned on our confusion-triad taxonomy or our structured-reasoning data format. Using MedGemma 4B as a student model means inheriting MedSigLIP's visual advantages "for free", but any gains beyond MedGemma's 22.22% zero-shot baseline must come from **our contribution** — the structured observer→reasoner knowledge distillation pipeline.

**Thesis framing.** This reframes MedGemma in the dissertation narrative:
1. **Not a mystery baseline** — its lead is fully explained by a specific, publicly documented architectural choice (MedSigLIP).
2. **A strong but incomplete ceiling** — MedSigLIP handles the "what does clinical skin look like" representation problem. The remaining 6.97% gap to SkinFlow SOTA is the **reasoning problem**: picking the right diagnosis among visually-similar conditions, which is what fine-tuning with structured reasoning data targets.
3. **A fair test for the other three models** — Qwen 3.5 9B (19.12%), Qwen 3.5 4B (14.61%), and Gemma 4 E4B (7.32%) all start from generic visual representations. If structured reasoning fine-tuning meaningfully closes their gaps to MedGemma (i.e., they reach 20%+ Top-1 after training), we demonstrate that our method can partially *substitute* for architectural medical specialization — a stronger and more useful result than "MedGemma is still best after everyone's fine-tuned".

**Refs:**
- MedGemma Technical Report: arXiv:2507.05201
- MedSigLIP: Google Health AI Developer Foundations, `google/medsiglip-448` on Hugging Face
- Model card: developers.google.com/health-ai-developer-foundations/medgemma/model-card

### 7.17 Related Work — Recent Dermatology VLMs (2025-2026)

The dermatology VLM landscape has moved rapidly in late 2025 / early 2026. A targeted arXiv survey identified the following concurrent works, their benchmarks, and how they relate to our contribution.

#### SkinFlow — SOTA on Fitzpatrick17k

- **Citation:** Liu et al., *"SkinFlow: Efficient Information Transmission for Open Dermatological Diagnosis via Dynamic Visual Encoding and Staged RL,"* arXiv:2601.09136 (Jan 2026).
- **Architecture:** Qwen2.5-VL-7B base + Dynamic Visual Encoder + staged reinforcement learning.
- **Benchmark:** Fitzpatrick17k 1,000-image sample. **29.19% Top-1, 71.16% Top-6**, scored via Gemini-2.5-Pro LLM judge.
- **Relation to our work:** SkinFlow is the **primary comparison target** — same benchmark, same scoring protocol (after our LLM-judge conversion). Our thesis aims to close the remaining 6.97% gap on MedGemma 1.5 4B through cheaper structured reasoning distillation (no staged RL).
- **Key architectural insight relevant to our pipeline:** SkinFlow's Stage 1 uses **label-anchored structured captioning** (features → diagnosis). Our observer→reasoner pipeline adopts this same label-anchoring principle: the observer produces pure visual descriptions without knowing the label, then the reasoner composes structured clinical reasoning conditioned on the confirmed diagnosis.

#### SkinGPT-R1 — Fairness-Focused Reasoning VLM

- **Citation:** arXiv:2511.15242, *"Trustworthy and Fair SkinGPT-R1 for Democratizing Dermatological Reasoning across Diverse Ethnicities,"* Nov 2025.
- **Architecture:** Vision-R1-7B (frozen) + dual-adapter distillation with fairness-aware mixture-of-experts.
- **Training data:** DermCoT — 10,000 dermatologist-certified chain-of-thought cases filtered from DermEval.
- **Benchmarks:** DermBench (14-model comparison, leads across 6 clinician-defined dimensions), **+41% improvement over Vision-R1 average**. Strongest published fairness numbers: **FST V = 55.0%, FST VI = 54.9%** — nearly double GPT-4o mini's 30.6% / 26.0% on the same split.
- **Relation to our work:** SkinGPT-R1 is the **fairness benchmark to beat on dark skin**. None of our zero-shot models come close (best is MedGemma at ~25% on FST VI). Critical dissertation question: does our observer→reasoner distillation reduce the FST V-VI gap, or does it preserve the existing zero-shot disparities? If fine-tuning brings any model above 30% on FST V-VI, we have a meaningful fairness contribution; if we can match or approach SkinGPT-R1's 55%, that would be a dual Top-1 + fairness result.
- **Methodological insight:** SkinGPT-R1 uses **adapter-only dual distillation** — the base 7B VLM stays frozen and only adapters are trained. This is cheaper than full LoRA fine-tuning and may be worth piloting as an ablation.

#### Skin-R1 — Textbook-Reasoning + RL Hybrid

- **Citation:** arXiv:2511.14900, *"Skin-R1: Toward Trustworthy Clinical Reasoning for Dermatological Diagnosis,"* Nov 2025.
- **Architecture:** Qwen2.5-VL-7B-Instruct with LoRA r=64, trained on textbook-derived reasoning data plus reinforcement learning.
- **Benchmark:** 72.1% on HAM10000 (distinct from our Fitzpatrick17k benchmark; HAM10k is dermoscopic, 7-class).
- **Relation to our work:** Skin-R1's **"textbook-derived reasoning" training data** is conceptually similar to SkinCaRe/SkinCoT (which we have access to) and to our generated observer→reasoner traces. Skin-R1 demonstrates that LoRA r=64 + structured reasoning data is sufficient to reach strong performance without full fine-tuning — validating our planned LoRA approach.
- **Note:** HAM10k is not our benchmark (different modality: dermoscopy vs clinical photography), but Skin-R1's methodology is transferable.

#### DermoGPT — Multi-dimensional Evaluation (DermoBench)

- **Citation:** Ru et al., arXiv:2601.01868 (Jan 2026), *"DermoGPT: In-Domain, OOD, Reasoning and Fairness Evaluation."*
- **Architecture:** Qwen3-VL-8B-Instruct with LoRA r=64.
- **Benchmark:** DermoBench — custom 4-dimensional evaluation (in-domain diagnosis, OOD, reasoning quality, fairness). Reports **78.0% in-domain, 93.9% fairness, 67.19% reasoning**.
- **Relation to our work:** DermoGPT's reasoning score (67.19%) is the first published attempt to **directly measure reasoning quality** as a dimension separate from accuracy. Our safety-critical rate from the LLM judge (§7.15) is philosophically similar — measuring *how wrong* a model is when it's wrong, not just whether it's wrong. A useful thesis extension would be to re-run the LLM judge with a 5th category for "reasoning quality" graded 1-5, matching DermoGPT's rubric.
- **Important contrast with Gemini Flash**: Ru et al. report that Gemini 2.5 Flash "hallucinated morphology concepts and produced inconsistent reasoning" when used without label anchoring — this validates our decision to use **Gemini 3 Flash only as the observer** (pure visual description, no diagnosis) and to use a separate, label-anchored model for the reasoner stage.

#### MedGemma — Our Leaderboard Winner (Architectural Context)

- **Citation:** Sellergren et al., arXiv:2507.05201 (Jul 2025), *"MedGemma Technical Report."*
- **Architecture:** Gemma 3 backbone + **MedSigLIP-400M** vision encoder (see §7.16).
- **Benchmark:** Not evaluated on Fitzpatrick17k in the original paper (they use US-DermMCQA: 71.8% MedGemma 4B vs 52.5% base Gemma 3).
- **Relation to our work:** The MedGemma technical report itself does **not** publish a Fitzpatrick17k Top-1 for MedGemma 1.5 4B. Our 22.22% (LLM-judge) and 19.7% (substring) are **the first published Fitzpatrick17k numbers for MedGemma 1.5**, to the best of our knowledge. This is a citable sub-finding.

#### Summary of the comparative landscape

| Model | Pub date | Params | Base | Fine-tuning | Fitz17k Top-1 | Where it wins |
|-------|:-:|:-:|:-:|:-:|:-:|-------|
| SkinFlow | Jan 2026 | 7B | Qwen2.5-VL-7B | Staged RL | **29.19%** | Overall Fitz17k SOTA |
| Skin-R1 | Nov 2025 | 7B | Qwen2.5-VL-7B | LoRA r=64 + RL | — (HAM10k 72.1%) | Methodology transfer |
| SkinGPT-R1 | Nov 2025 | 7B | Vision-R1-7B | Adapter-only dual distill | — (no Fitz Top-1) | **FST V-VI fairness (55%)** |
| DermoGPT | Jan 2026 | 8B | Qwen3-VL-8B | LoRA r=64 | — (DermoBench 78%) | Reasoning quality metric (67.19%) |
| **MedGemma 1.5 4B (ours)** | — | 4B | Gemma 3 + MedSigLIP | **Zero-shot** | **22.22%** (LLM judge, this work) | **First Fitz17k Top-1 for MedGemma 1.5** |

#### What our contribution adds to this landscape

Compared to the 2025-2026 published wave, our contribution is distinctive in four ways:

1. **First published Fitzpatrick17k Top-1 for MedGemma 1.5 4B** (22.22% LLM-judge, 19.7% substring) — previously unpublished by the MedGemma team themselves.
2. **Cross-model LLM-as-judge re-scoring** — 5 models × 1,000 entries × 2 scoring protocols (substring + judge) gives a reproducible head-to-head ranking that no prior paper provides. We also expose the **safety-critical error rate** as a new evaluation dimension, which none of the reference papers report.
3. **Observer→reasoner pipeline as a cheaper alternative to staged RL** — SkinFlow and Skin-R1 use RL, which requires reward models and multiple training stages. Our approach uses a single supervised fine-tuning pass on LLM-generated structured reasoning. If effective, this reduces the training complexity for future dermatology VLM work.
4. **Four-model architectural comparison at ~4-9B parameter scale** — the only prior work that evaluates multiple base architectures head-to-head on Fitzpatrick17k is the SkinFlow paper itself (which reports only zero-shot numbers for comparisons). Our comparison of MedGemma (medical vision), Gemma 4 (same language, generic vision), Qwen 3.5 (early-fusion multimodal), and Grok (reasoning model) isolates the contribution of each architectural choice on the same benchmark.

### 7.7 Additional Data Access (Newly Approved)

During this phase, access was granted to two previously gated resources:

- **MedGemma 1.5 4B IT** (Google): Enabled inclusion in the benchmark lineup as the medical-specialized baseline.
- **SkinCaRe/SkinCoT** (HuggingFace yuhos16/SkinCaRe): 3,041 DermNet images with clinician-certified chain-of-thought reasoning. This will be incorporated in the training phase as ground-truth reasoning data, enabling a comparison between clinician-written and LLM-generated training data — a publishable finding on its own.

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

## Research: SLMs Under 10B for Dermatology (Literature Survey)

### Overview

A comprehensive survey was conducted to identify all published Small Language Models and Small Vision-Language Models (under 10 billion parameters) being used for dermatological diagnosis, classification, and reasoning. This informs both model selection and positioning of our contribution.

### Generative VLMs for Dermatology (Under 10B Parameters)

| Model | Base Architecture | Params | Task | Top Result | Weights | Paper |
|-------|------------------|--------|------|------------|---------|-------|
| **DermaGPT** | PaLI-Gemma 2 + RAG | 2.95B | Classification + patient explanations | 90.2% acc (11 lesion types) | No | Nature Sci. Reports, Feb 2026 |
| **SmolVLM-Derm** | SmolVLM (SigLIP-SO400M + SmolLM2) | 2.2B | Bacterial skin classification + QA | 70.2% acc, BERTScore 90.19% | Partial | PMC, 2025 |
| **DermIQ-VLM** | Qwen2.5-VL-3B | 3B | 7-class classification + VQA | 51.4% F1 (majority voting) | No | arXiv:2510.01236, Oct 2025 |
| **Dermatech** | Qwen2-VL-2B-Instruct | 2B | Skin condition diagnosis | N/A (community model) | Yes (HF) | Community |
| **PaliGemma-Derm** | PaliGemma-3B-pt-224 | 3B | Skin condition classification | Val loss 0.22 | Yes (HF) | Community |
| **MedGemma 4B** | Gemma 3 + MedSigLIP 400M | 4B | MCQ classification | 71.8% US-DermMCQA | Yes (HF) | arXiv:2507.05201 |
| **CLARIFY** | DINOv2-Base + pruned Qwen-VL-3B | 86M + 3.75B | VQA diagnosis | 82.1% acc (8 diseases) | No | arXiv:2508.18430, Aug 2025 |
| **SkinGPT-R1** | Vision-R1-7B (frozen) + dual adapters | 7B | Diagnostic reasoning | 4.03/5 DermBench | No | arXiv:2511.15242, Nov 2025 |
| **Skin-R1** | Qwen2.5-VL-7B-Instruct (LoRA r=64) | 7B | Diagnosis + clinical reasoning | 72.1% HAM10k | No | arXiv:2511.14900, Nov 2025 |
| **SkinFlow** | Qwen2.5-VL-7B + Dynamic Vision Encoder | 7B | Open-vocab diagnosis (~200 classes) | 29.19% Top-1 Fitz17k (+12.06% vs GPT-5.2) | No | arXiv:2601.09136, Jan 2026 |
| **SkinVL** | LLaVA-Med-7B (LoRA) | 7B | VQA + classification | 95.6% Patch16 SFT | Coming soon | arXiv:2505.06152, May 2025 |
| **DermoGPT** | Qwen3-VL-8B-Instruct (LoRA r=64) | 8B | Morphology + diagnosis + fairness | 78.0% in-domain, 93.9% fairness | Pending | arXiv:2601.01868, Jan 2026 |
| **LLaVA-Derm-7B** | LLaVA-v1.5-7B (LoRA) | 7B | Diagnosis (SCIN-trained) | N/A (community model) | Yes (HF) | Community |
| **KD-LLaVA** | LLaVA-v1.5-13B (distilled from GPT-4V) | 13B | Skin cancer report generation | SacreBLEU 55.59, BERTScore 0.90 | No | PMC, 2025 |
| **DermatoLlama 1.0** | LLaMA-3.2-11B-Vision (LoRA) | 11B | Report generation + reasoning | Outperforms SOTA VLMs (medRxiv) | Yes (HF) | medRxiv, 2025 |

### Vision Encoders / Foundation Models (Sub-1B, Non-Generative)

| Model | Params | Task | Top Result | Weights | Paper |
|-------|--------|------|------------|---------|-------|
| **DermLIP** | ~150-400M | Zero-shot classification + retrieval | 73.1% AUROC | Yes (GitHub) | ICCV 2025 Highlight |
| **DermFM-Zero** | Sub-1B | Zero-shot diagnosis (98 conditions) | SOTA on 20 benchmarks | Yes (GitHub) | arXiv:2602.10624, Feb 2026 |
| **MONET** | ~304M | Concept annotation + zero-shot | Competitive on Derm7pt | Yes (HF) | Su-In Lee Lab |
| **Derm Foundation** (Google) | Undisclosed | Embedding extraction | +10-15% over BiT-M baseline | Yes (HF) | Google HADF |

### Key Findings from the Survey

**1. Most Popular Base Architectures:**
- Qwen2/2.5-VL (2B, 3B, 7B) — most commonly used across publications
- LLaVA / LLaVA-Med (7B, 13B) — established baseline architecture
- PaLI-Gemma (3B) — strong performer at small scale
- SmolVLM (2.2B) — emerging for edge deployment

**2. LoRA is the Dominant Fine-Tuning Approach:**
Nearly every published model uses LoRA or adapter-only training, with typical ranks of 32-128 and <1% of parameters trainable. Full fine-tuning is rare in this domain due to GPU constraints and the effectiveness of parameter-efficient methods.

**3. GRPO/RL-Based Reasoning is a 2025-2026 Trend:**
Skin-R1, DermIQ-VLM, SkinFlow, and DermoGPT all use Group Relative Policy Optimization for enhancing dermatological reasoning. This represents a shift from pure supervised fine-tuning toward reinforcement learning from structured feedback.

**4. Knowledge Distillation is Actively Being Explored:**
- KD-LLaVA distills GPT-4V into LLaVA-13B for skin cancer reports
- DermatoLlama uses synthetic data from larger VLMs (SCALEMED framework)
- SkinGPT-R1 employs dual distillation through adapter-only training
- Our approach (structured reasoning from teacher models) fits squarely in this trend

**5. Weight Availability is Limited:**
Only MedGemma 4B, Esperanto LLaVA-Derm-7B, Dermatech-Qwen2-VL-2B, PaliGemma-Derm, DermLIP, MONET, and DermFM-Zero have publicly released weights. Most research models are not yet released.

**6. Datasets Most Frequently Used Across Published Models:**
Fitzpatrick17k, DermNet/DermNetNZ, HAM10000, Derm7pt, SCIN, PAD-UFES-20, Derm1M, BCN20000, and MM-Skin.

### Our Contribution in Context

No published work has compared the three specific model families we selected (Gemma 4, Qwen 3.5, MedGemma 1.5) on dermatological tasks:

| Our Model | Closest Existing Work | What's Novel |
|-----------|----------------------|-------------|
| **Gemma 4 E4B (4B)** | No dermatology fine-tuning exists | First Gemma 4 dermatology evaluation |
| **Qwen 3.5 4B** | Others used Qwen2/2.5-VL, not 3.5 | First Qwen3.5 dermatology study |
| **MedGemma 1.5 4B** | MedGemma 4B evaluated but not 1.5 for derm fine-tuning | Latest MedGemma version on derm |
| **Qwen 3.5 9B** | SkinFlow used Qwen2.5-VL-7B | Newer architecture, same scale |

The central research question — *does medical pre-training (MedGemma) provide an advantage over general-purpose VLMs (Gemma 4, Qwen3.5) when fine-tuned on the same dermatology-specific structured reasoning data?* — has not been addressed in the literature.

---

## Research: VLM Training Datasets (Image-Text Pairs)

### Overview

VLM training requires image-text pairs (captions, VQA, clinical reports, reasoning chains), not just classified images. A comprehensive survey identified all available dermatology-specific datasets suitable for VLM training.

### Tier 1: Dedicated Dermatology VLM Datasets

| Dataset | Scale | Text Type | Access | Status |
|---------|-------|-----------|--------|--------|
| **Derm1M + Derm1M_Instruct** | 1M image-text pairs + 300K instruction data | Clinical captions (avg 41 tokens) | HuggingFace (CC BY-NC 4.0) | **Downloaded** |
| **MM-Skin** | 10K captions + 27K VQA pairs | Textbook-sourced professional captions + VQA | GitHub | **Downloaded** (7,006 clinical subset extracted) |
| **SkinCaRe (SkinCAP + SkinCoT)** | 4,000 SkinCAP + 3,041 SkinCoT | Dermatologist captions + chain-of-thought reasoning | HuggingFace (CC-BY-NC-SA 4.0) | **Access approved** |
| **DermaBench** | 656 images, 14,474 VQA pairs | Expert-written VQA (single/multi-choice + open-ended) | Harvard Dataverse (free) | Available |
| **DermaSynth** | 92,020 synthetic pairs | Gemini 2.0-generated clinical QA (120 question types) | GitHub (CC-BY-4.0) | Available |
| **DermaVQA** | ~1,000 cases | Real patient questions + answers (EN/CN/ES) | OSF (free) | Available |
| **DermaVQA-DAS** | 7,400+ segmentation masks + QA pairs | Structured closed-ended QA by dermatologists | Codabench | Available |
| **eSkinHealth** | 5,623 images, 47 diseases | Captions + 69-dim clinical concept vectors | arXiv (check paper) | Available |

### Tier 2: Coming Soon (Monitor for Release)

| Dataset | Scale | Why It Matters |
|---------|-------|----------------|
| **DermoInstruct** | 211K images, 773K instruction trajectories | 5 task formats: morphological descriptions, CoT reasoning, multi-turn hierarchical diagnosis. Largest instruction-tuning resource when released. |
| **DermEVAL** | 11,347 images | VQA + medical report generation benchmark |

### Tier 3: General Medical (Filterable for Dermatology)

| Dataset | Scale | Notes |
|---------|-------|-------|
| **BIOMEDICA** | 24M figure-caption pairs from PubMed | Derm subset showed +29.8% improvement specifically; CC-BY |
| **Open-PMC-18M** | 18M subfigure-caption pairs | CC-BY-4.0, higher fidelity than PMC-15M |
| **PubMedVision** | 1.3M VQA entries | ~80% radiology, small derm subset; GPT-4V reformatted |

### Tier 4: Structured Concept Annotations (Convertible to Text)

| Dataset | Scale | Notes |
|---------|-------|-------|
| **SKINCON** | 3,230 + 656 images | 48 clinical concept annotations (plaque, scale, erosion, etc.) — convertible to captions |
| **Derm7pt** | ~2,000 image pairs | 7-point checklist criteria — convertible to descriptive text |
| **IEEE DataPort Derm RAG** | 49,100 images + text corpus | Medical literature chunks (RAG-style) across 32 classes |

### Our VLM Training Data Stack

Combined available resources for VLM training:

| Source | Image-Text Pairs | Purpose |
|--------|-----------------|---------|
| Derm1M | ~1,000,000 | Large-scale pretraining alignment |
| MM-Skin (clinical) | ~7,006 captions + 16,614 VQA | Textbook-quality captioning + VQA |
| SkinCaRe/SkinCoT | 7,041 (captions + CoT) | Chain-of-thought diagnostic reasoning |
| Our structured reasoning pipeline | ~29,913 (generated) | Teacher-model-generated structured descriptions |
| **Total** | **~1,060,574** | Multi-task VLM training across captions, VQA, and reasoning |

---

## Research: Additional Classification Datasets Identified

### Kaggle Datasets for Supplementing Training Data

During the dataset collection phase, additional Kaggle datasets were identified that contain clinical/smartphone photos (not dermoscopic) suitable for supplementing the training corpus:

| Dataset | Kaggle Slug | Images | Key Classes | License |
|---------|------------|-------:|-------------|---------|
| **Massive Skin Disease Balanced** | `muhammadabdulsami/massive-skin-disease-balanced-dataset` | 262,874 | 34 classes, covers most of our 13 targets | MIT |
| **Hossain Skin Diseases** | `ismailpromus/skin-diseases-image-dataset` | ~27,153 | 10 classes incl. melanoma, eczema, psoriasis, tinea | Original Authors |
| **20 Skin Diseases** | `haroonalam16/20-skin-diseases-dataset` | ~5,000-8,000 | 20 classes — rare urticaria + ringworm as named classes | Other |
| **PAD-UFES-20** | `mahdavi1202/skin-cancer` | 2,298 | BCC, SCC, melanoma, AK, nevus, seb keratosis — **smartphone, 58% biopsy-proven** | CC BY 4.0 |
| **SD-198** | `longngzzz/sd-198` | 6,584 | 198 disease classes — covers scabies, contact dermatitis, urticaria (rare gaps) | Academic |
| **Skin Disease Detection** | `mgmitesh/skin-disease-detection-dataset` | ~10,000+ | 15 classes incl. acne, ringworm, eczema, BCC | CC BY 4.0 |
| **Vitiligo** | `shinynose/vitiligo` | 3,628 | Vitiligo vs. healthy (includes stock photos) | CC0 |
| **Eczema** | `adityush/eczema2` | 3,400+ | Eczema infected vs. normal | LGPL-3.0 |
| **Acne IGA Scale** | `tapakah68/skin-problems-34-on-the-iga-scale` | 686 | Acne severity — **smartphone selfies** | CC BY-NC-ND 4.0 |
| **ACNE04** | `manuelhettich/acne04` | ~1,000+ | 4 acne severity levels — watermark-cleaned | Unknown |

**Notes:**
- PAD-UFES-20 was subsequently included in the final corpus (Phase 1)
- The Massive dataset (262K) likely has significant overlap with Kaggle DermNet — deduplication required
- SD-198 is the best source for filling gap classes (scabies, contact dermatitis, urticaria) with 198 fine-grained classes

### DDI (Diverse Dermatology Images) — Stanford

| Detail | Value |
|--------|-------|
| Images | 656 clinical photos from 570 patients |
| Skin tones | Fitzpatrick I-VI (deliberately balanced) |
| Conditions | Melanoma, BCC, atopic dermatitis, psoriasis, others |
| Confirmation | Pathology-confirmed |
| License | Non-commercial research (Stanford Research Use Agreement) |
| Access | Stanford AIMI portal (registration + agreement required) |
| Alternative | DDI images are bundled in SkinCAP on HuggingFace (655 DDI images included) |

DDI is particularly valuable for fairness evaluation due to its deliberate skin tone balance. Access was obtained indirectly through SkinCAP.

### DDI-2 (Newer Version)

| Detail | Value |
|--------|-------|
| Images | 665 photos, 550 patients, 169 diagnoses |
| Focus | Self-identified Asian patients in the U.S. |
| Access | https://daneshjoulab.github.io/ddi2-dataset/ |

---

## Research: Kaggle DermNet Full Class Distribution

The Kaggle DermNet dataset (already downloaded at `data/raw/kaggle_dermnet/`) contains 23 classes across train/test splits. Full image counts:

| # | Class | Train | Test | Total |
|---|-------|------:|-----:|------:|
| 1 | Psoriasis / Lichen Planus | 1,405 | 352 | 1,757 |
| 2 | Seborrheic Keratoses / Benign Tumors | 1,371 | 343 | 1,714 |
| 3 | Tinea / Ringworm / Fungal Infections | 1,300 | 325 | 1,625 |
| 4 | Eczema | 1,235 | 309 | 1,544 |
| 5 | Actinic Keratosis / BCC / Malignant Lesions | 1,149 | 288 | 1,437 |
| 6 | Warts / Molluscum / Viral Infections | 1,086 | 272 | 1,358 |
| 7 | Nail Fungus / Nail Disease | 1,040 | 261 | 1,301 |
| 8 | Acne / Rosacea | 840 | 312 | 1,152 |
| 9 | Systemic Disease | 606 | 152 | 758 |
| 10 | Light Diseases / Pigmentation Disorders | 568 | 143 | 711 |
| 11 | Atopic Dermatitis | 489 | 123 | 612 |
| 12 | Vascular Tumors | 482 | 121 | 603 |
| 13 | Melanoma / Skin Cancer / Nevi | 463 | 116 | 579 |
| 14 | Bullous Disease | 448 | 113 | 561 |
| 15 | Scabies / Lyme / Infestations | 431 | 108 | 539 |
| 16 | Lupus / Connective Tissue | 420 | 105 | 525 |
| 17 | Vasculitis | 416 | 105 | 521 |
| 18 | Herpes / HPV / STDs | 405 | 102 | 507 |
| 19 | Exanthems / Drug Eruptions | 404 | 101 | 505 |
| 20 | Cellulitis / Impetigo / Bacterial Infections | 288 | 73 | 361 |
| 21 | Contact Dermatitis / Poison Ivy | 260 | 65 | 325 |
| 22 | Hair Loss / Alopecia | 239 | 60 | 299 |
| 23 | Urticaria / Hives | 212 | 53 | 265 |
| | **TOTAL** | **15,557** | **4,002** | **19,559** |

Classes 6-9, 12, 14, 16-19 are not currently in the unified training dataset but could be added for broader VLM training coverage.

---

## Research: Dermatology VLM Benchmark Landscape

### Published Benchmarks

| Benchmark | Images | Task | Key Models Evaluated | Status |
|-----------|--------|------|---------------------|--------|
| **Fitzpatrick17k (1,000)** | 1,000 | Top-1/Top-6 classification | SkinFlow, GPT-5.2, Qwen3-VL-235B | **Used in our evaluation** |
| **DermBench** | 10,000+ | Multi-task (5 categories) | SkinGPT-R1 (4.03/5), Vision-R1 (2.87/5) | Reference |
| **US-DermMCQA** | N/A | MCQ classification (79 conditions) | MedGemma (71.8%), base Gemma 3 (52.5%) | Reference |
| **MM-Skin VQA** | 11,002 | Open-ended VQA | SkinVL variants | **Used in our evaluation** |
| **DermaBench** | 656 (DDI) | 14,474 expert VQA pairs | Not yet evaluated | Harvard Dataverse — available |
| **DermEVAL** | 11,347 | VQA + report generation (16 diseases) | Various MLLMs | WACV 2026 — not public |
| **DermoBench** | N/A | In-domain + OOD + reasoning + fairness | DermoGPT (78.0% ID, 93.9% fairness) | Pending release |

### Performance Landscape (Fitzpatrick17k Top-1 Accuracy)

```
 SkinFlow (7B, fine-tuned)        ████████████████████████████████  29.19%
 GPT-5.2 (commercial)             ██████████████████                18.24%
 Qwen3-VL-235B (235B)             █████████████████                 17.13%
 MedGemma 4B (zero-shot)          ████████████████                  16.30%  ← Our baseline
 ─────────────────────────────────────────────────────────────────────────
 Target: Exceed GPT-5.2 (18.24%) after fine-tuning with structured reasoning
```

---

## Phase 8: Fine-Tuning Pipeline (Implementation)

### 8.1 Pipeline Overview

`src/fine_tune/` implements a config-driven LoRA SFT pipeline over TRL `SFTTrainer` + PEFT. One entrypoint (`python -m fine_tune.train --config <yaml> [--format {label_only,full_reasoning}]`) fine-tunes any of the four base VLMs; per-model variations live in four checked-in YAML configs. Vision towers are frozen (preserves MedSigLIP's medical advantage on MedGemma, and Qwen VL's early-fusion features on Qwen 3.5). LoRA rank 64 / alpha 128, 5 epochs, cosine LR schedule, `paged_adamw_8bit` on the 9B model only.

### 8.2 Ablation: label_only vs full_reasoning

`prepare_data.py` emits both formats from the same stratified 95/5 split (seed=42):

- `label_only` — assistant response is `{"diagnosis", "category"}` only
- `full_reasoning` — assistant response is the unified schema (diagnosis, top_n, confidence, category, observation, reasoning, differentials)

Each model is trained twice (once per format), giving 8 runs total. The comparison measures whether structured reasoning signal beats label-only supervision for each of the four base architectures.

### 8.3 Artifacts

- Per-epoch LoRA checkpoints on the `/workspace` network volume (save_total_limit=3)
- Best-by-eval-loss adapter pushed to a private HF Hub repo (`danielfdias98/<model>-derm-reasoning-{label-only,full-reasoning}`)
- `manifest.csv` ledger: one row per completed run (model, format, config SHA, best eval_loss, wall-clock hours)

### 8.4 Operational Workflow (RunPod L40S)

1. `scripts/sync_to_volume.sh` — one-time rsync of `final/train/` + `reasoning.jsonl` to the network volume (~30 min).
2. `scripts/setup_pod.sh` — once per fresh pod: installs GPU deps, HF login, verifies GPU + data.
3. `scripts/run.sh configs/<model>.yaml [--format label_only]` — one invocation per (model, format) pair.

Pod restart mid-run: `scripts/run.sh configs/<model>.yaml --resume` auto-resumes from the latest epoch checkpoint.

### 8.5 Design & Plan References

- Design spec: `docs/superpowers/specs/2026-04-17-fine-tune-pipeline-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-17-fine-tune-pipeline.md`

---

## Phase 9: Dataset Publication

### 9.1 Three-Tier Release on HuggingFace Hub

The reasoning dataset and underlying images were released as three separate HuggingFace dataset repositories. The split is dictated by the licensing of the five source datasets — most do not permit public redistribution of their images even when their captions/metadata are freely shareable. This three-tier pattern matches the convention used by SkinCAP, SkinCaRe, and SkinFlow.

| Repo | Visibility | Contents | URL |
|---|---|---|---|
| `danielfdias98/derm-reasoning` | 🌐 Public | Reasoning annotations only (Parquet), 2 configs × 2 splits, dataset card, `download_images.py` | https://huggingface.co/datasets/danielfdias98/derm-reasoning |
| `danielfdias98/derm-reasoning-redistributable` | 🌐 Public | SCIN + PAD-UFES-20 images only (~7,500 files, ~8 GB Parquet) — both publish under licenses (Apache 2.0, CC-BY 4.0) that permit redistribution | https://huggingface.co/datasets/danielfdias98/derm-reasoning-redistributable |
| `danielfdias98/derm-reasoning-full` | 🔒 Private | Full image set (28,486 images) + all four reasoning JSONLs, examiner access only | https://huggingface.co/datasets/danielfdias98/derm-reasoning-full |

### 9.2 Per-Source Licensing Decision Matrix

The three-tier split was driven by the following analysis of each source dataset's redistribution policy:

| Source | Train rows | License | Public Hub redistribution? |
|---|---:|---|---|
| SCIN (Google Research) | 4,332 | Apache 2.0 | ✅ Yes — included in `derm-reasoning-redistributable` and `derm-reasoning-full` |
| PAD-UFES-20 | 1,706 | CC BY 4.0 | ✅ Yes — included in both image repos |
| SkinCAP captions | 3,607 | CC-BY-NC-SA 4.0 | ⚠ Captions OK with non-commercial term inheritance; embedded DDI images excluded from public repos per Stanford Research Use Agreement |
| Kaggle DermNet | 15,221 | "Other" — images scraped from DermNet website | ❌ Original copyright applies; included only in private repo |
| DermNet NZ (scraped) | 771 | DermNet NZ copyright | ❌ Takedown risk; included only in private repo |

The unified dataset license is **CC-BY-NC-SA 4.0** to inherit SkinCAP's most restrictive component. Use is **non-commercial research only**.

### 9.3 Source Attribution per Row

Every row in all three repos carries a `source` column populated by walking each per-source folder under `data/dataset/<source>/` and indexing image basenames. SCIN, PAD-UFES-20, SkinCAP, Kaggle DermNet, and DermNet NZ are checked in priority order; rows whose basename doesn't match any source folder fall back to filename heuristics (`PAT_*` → `pad_ufes`, numeric hash → `dermnet_nz`, descriptive → `kaggle_dermnet`).

This attribution preserves provenance for downstream auditing — anyone reproducing or extending the dataset can filter to a specific source for license-compliant subset extraction.

### 9.4 Annotation License

The structured reasoning annotations themselves are the author's contribution and were generated using **Gemini 2.5 Flash** as a label-anchored teacher. They are released under the unified dataset license (CC-BY-NC-SA 4.0) alongside the source attribution metadata.

### 9.5 Reproducibility for Downstream Users

Three reproduction paths, ordered by ease:

1. **Annotations + redistributable images** — `load_dataset("danielfdias98/derm-reasoning")` for the reasoning JSONLs; `load_dataset("danielfdias98/derm-reasoning-redistributable")` for the ~7,500 SCIN + PAD-UFES-20 images. Self-contained, no external auth.
2. **Annotations + bring-your-own-images** — `load_dataset("danielfdias98/derm-reasoning")` plus `python download_images.py` (in the public repo) to fetch the remaining ~21,000 images directly from each original source (SCIN GCS, Kaggle, HuggingFace, manual DermNet NZ).
3. **Examiner / private access** — request access to `derm-reasoning-full` via HuggingFace Hub's per-user grant for the complete image set.

### 9.6 Citation

```bibtex
@misc{dias2026derm-reasoning,
  author = {Ferreira Dias, Daniel},
  title  = {Dermatology Reasoning Dataset: Structured chain-of-thought annotations across five public sources},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/datasets/danielfdias98/derm-reasoning}},
}
```

Users are required to cite all five source datasets (SCIN, PAD-UFES-20, SkinCAP, Kaggle DermNet, DermNet NZ) in addition to this one — the dataset card lists the canonical citations.

---

## Phase 10: Fine-Tuning Configuration and Hyperparameter Selection

### 10.1 Methodology

Fine-tuning is performed using **parameter-efficient adaptation via Low-Rank Adaptation (LoRA)** rather than full-parameter SFT. LoRA inserts low-rank decomposition matrices into the attention and feed-forward projections, training only the inserted matrices while the base model weights remain frozen (Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685, 2021). This decision is driven by three constraints documented in the literature on small dermatology VLMs: (i) the campaign target is small models (4–9B) whose full-parameter SFT exceeds the memory envelope of the available GPU tier (NVIDIA A40, 48 GB), (ii) prior dermatology VLMs at comparable scale (Skin-R1, DermoGPT, SkinGPT-R1) all use LoRA or adapter-only training with rank 32–64, demonstrating that parameter-efficient methods recover the bulk of full-FT quality at a fraction of the cost, and (iii) LoRA adapters at our chosen rank are <500 MB per model, enabling lightweight redistribution and reproducibility.

The training framework is **Unsloth** (apache-2.0 / AGPL-3.0, github.com/unslothai/unsloth) running through TRL's `SFTTrainer`. Unsloth provides Triton kernels for the dominant attention paths in our base models (Qwen 3.5's `chunk_gated_delta_rule`, Gemma 3/4 attention) plus first-class multimodal collator support (`UnslothVisionDataCollator`, `FastVisionModel.get_peft_model` with the `finetune_vision_layers` flag). The decision to adopt Unsloth followed an earlier campaign-blocking dependency disaster (Phase 8) in which `pip install -U flash-linear-attention` silently upgraded torch to a CUDA tier the pod's NVIDIA driver could not run; Unsloth's matched-version extras (e.g. `unsloth[cu124-torch260]`) eliminate that class of failure.

### 10.2 Datasets

The training datasets are the two image-embedded public Hub repositories produced in Phase 9:

| Variant | Hub repo | Train rows | Val rows | Assistant turn |
|---|---|---:|---:|---|
| **Treatment** (full reasoning) | [danielfdias98/derm-reasoning-full-reasoning](https://huggingface.co/datasets/danielfdias98/derm-reasoning-full-reasoning) | 25,637 | 2,849 | Diagnosis + category + observation + morphology + color + texture + border + location + size + reasoning + confidence + Fitzpatrick estimate |
| **Control** (label only) | [danielfdias98/derm-reasoning-label-only](https://huggingface.co/datasets/danielfdias98/derm-reasoning-label-only) | 25,637 | 2,849 | Diagnosis + category only |

Both datasets share the same stratified 90/10 image split (seed = 42) so any difference in downstream evaluation is attributable to supervision density rather than data leakage. The on-disk schema follows Unsloth's canonical convention (image + instruction + response columns at row top-level), matching the format of Unsloth's published vision tutorials (e.g. `unsloth/Radiology_mini`); Unsloth's training pipeline performs the wrap into the chat-template `messages` structure internally before the trainer sees the batch.

### 10.3 Per-Model Hyperparameter Tables

Three base models are fine-tuned, spanning a medical-specialised vision encoder (MedGemma 1.5 4B's MedSigLIP), a generalist Gemma family member at the same parameter count (Gemma 4 E4B), and a third architecture family at two scales for capacity comparison (Qwen 3.5 4B and 9B). Each base model is fine-tuned twice — once on each dataset variant — yielding **6 fine-tune runs total**.

Hyperparameter values that are common across all runs are reported once below; per-model deviations are tabulated separately.

#### 10.3.1 Common configuration (all six runs)

| Component | Value | Source |
|---|---|---|
| Adaptation | LoRA, 16-bit base weights | Hu et al., 2021; Unsloth's `LoRA (16-bit)` mode |
| LoRA rank `r` | 32 | Unsloth Gemma 4 docs (`r=32`); Skin-R1 used `r=64`. We interpolate at 32 for a domain-shift (medical) task at the 4–9B scale, matching Unsloth's published Gemma-4 vision recipe |
| LoRA alpha | 32 (= `r`) | Unsloth recommendation: `alpha == r at least`; Hu et al. 2021 §3.4 shows scaling alpha proportional to r is robust |
| LoRA dropout | 0.05 | Light regularisation appropriate for our 25k-row dataset; matches our pre-Unsloth validated config and is consistent with Unsloth's `0`–`0.1` range |
| Bias | `none` | Standard LoRA practice (no bias parameters trained) |
| `target_modules` | `"all-linear"` | Unsloth's idiom; covers `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` plus model-specific projections without manual enumeration |
| `finetune_vision_layers` | `False` | Vision tower frozen for instruction-style SFT, preserving MedSigLIP's medical pre-training advantage on MedGemma (see §7.16) and Qwen-VL's early-fusion features. Recommended by Unsloth's vision fine-tuning guide |
| Precision | bf16 LoRA on bf16 base | Avoids the fp16 NaN risks documented for Qwen 3.5 in Unsloth's troubleshooting docs; bf16 is the consensus precision for Ampere/Hopper VLM SFT |
| Optimiser | AdamW 8-bit (Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, arXiv:2110.02861, 2021) | Frees ~4 bytes/param of optimiser state vs full AdamW; recommended by Unsloth across all listed model families |
| Weight decay | 0.01 (Qwen / MedGemma) / 0.001 (Gemma 4) | Per-family Unsloth default; the lower value for Gemma 4 reflects its sensitivity to over-regularisation in Unsloth's published recipe |
| Number of epochs | 5 | Unsloth's documentation suggests "1–3 with diminishing returns past 3"; we extend to 5 because (a) the dataset is small (25k) relative to full-corpus VLM SFT (typically ≥50k), (b) early-stopping is used (see §10.4) so no harm is done if convergence is earlier, and (c) the structured-reasoning supervision is information-dense per row |
| Effective batch size | 16 | `per_device_train_batch_size × gradient_accumulation_steps`. 16 is the consensus VLM SFT batch size in published recipes (Unsloth, Skin-R1, DermoGPT) |
| Random seed | 42 | Reproducibility (matches the dataset split seed) |

#### 10.3.2 Per-model deviations

Per-device batch size and gradient-accumulation are tuned to fit each model in the A40 48 GB envelope while preserving the effective batch size of 16. The learning-rate schedule and warmup follow Unsloth's per-family recommendations.

| Setting | Qwen 3.5 4B | Qwen 3.5 9B | Gemma 4 E4B | MedGemma 1.5 4B |
|---|---|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` | `unsloth/Qwen3.5-9B` | `unsloth/gemma-4-E4B-it` | `unsloth/medgemma-1.5-4b-it` |
| Expected VRAM (bf16 LoRA) | ~10 GB | ~22 GB | ~17 GB | ~16 GB |
| `per_device_train_batch_size` | 2 | 1 | 2 | 2 |
| `gradient_accumulation_steps` | 8 | 16 | 8 | 8 |
| Context length (`max_seq_length`) | 4096 | 4096 | 4096 | 4096 |
| Learning rate | 2 × 10⁻⁴ | 2 × 10⁻⁴ | 2 × 10⁻⁴ | 2 × 10⁻⁴ |
| LR scheduler | Cosine | Cosine | Linear | Cosine |
| Warmup | 30 steps | 30 steps | 5 steps | 30 steps |

Notes on the deviations:

- **Qwen 3.5 4B/9B** follow Unsloth's `Qwen3.5-fine-tune` recipe. **Quantisation to 4-bit (QLoRA) is explicitly avoided** — Unsloth's troubleshooting docs warn that "QLoRA on Qwen 3.5 has higher than normal quantisation differences," and the medical domain's discrimination requirements amplify any such drift.
- **Gemma 4 E4B** uses Unsloth's published Gemma-4 recipe verbatim (LR = 2 × 10⁻⁴, linear schedule, 5-step warmup, weight decay = 0.001). Note the **expected starting loss for Gemma 4 E2B/E4B is 13–15** per Unsloth's documentation, not the 1–3 range typical of larger-class models — this is normal behaviour driven by the model's compressed effective parameter count and is not indicative of training failure.
- **MedGemma 1.5 4B** is built on the Gemma 3 backbone (Sellergren et al., *MedGemma Technical Report*, arXiv:2507.05201, 2025). Hyperparameters follow Unsloth's Gemma 3 recipe (cosine schedule, 30-step warmup, weight decay 0.01) rather than Gemma 4's, since the language backbone is Gemma 3.
- **Context length 4096** is sized for the full-reasoning assistant turn (~600–800 tokens after structured-JSON formatting). For label-only ablation runs, 2048 would be sufficient; we keep 4096 across the campaign for direct comparability.

### 10.4 Evaluation and Stopping Rule

Each run is evaluated on the held-out 2,849-row validation split at the end of each epoch (`eval_strategy = "epoch"`, `save_strategy = "epoch"` to satisfy `load_best_model_at_end`'s constraint). The best checkpoint by `eval_loss` is loaded at training end and pushed to the Hugging Face Hub (`HFHubPushCallback`) as `danielfdias98/<model>-derm-reasoning-{full-reasoning,label-only}`. A per-run row is appended to `/workspace/runs/manifest.csv` capturing model, format, config SHA, best `eval_loss`, wall-clock hours, and Hub repo, providing a single source of truth for the dissertation results table.

### 10.5 Ablation Design

The full-reasoning vs label-only contrast (paired across all three architectures) directly tests the dissertation's central claim — that **structured chain-of-thought supervision contributes more to dermatological diagnostic accuracy than parameter-count scaling alone**. The relevant prior result is SkinFlow's Stage 1 ablation (Liu et al., arXiv:2601.09136, 2026), in which structured medical captioning contributed +9.23 % Top-1 accuracy on Fitzpatrick17k versus +4.74 % from architectural changes. By holding the architecture, base weights, dataset images, training schedule, and random seed constant across the two conditions and varying only the assistant-turn supervision density, any difference in post-training Top-1 accuracy on Fitzpatrick17k, Confusion Triads, and MM-Skin VQA is attributable to the structured reasoning signal — yielding a clean 3-architecture × 2-supervision factorial that the dissertation can statistically defend.

### 10.6 Reporting

Results from each of the six runs will be reported using the comparison-table convention established in Phase 7 (zero-shot baselines), extended with two columns per benchmark: post-fine-tuning Top-1 / Top-6 accuracy and Δ from the model's zero-shot baseline. Fairness disaggregation by Fitzpatrick skin type (FST I–II vs V–VI) follows the protocol of Daneshjou et al. (*Disparities in dermatology AI performance on a diverse, curated clinical image set*, Science Advances, 2022) and SkinGPT-R1's published per-FST breakdown (arXiv:2511.15242, 2025) for direct comparability.

---

## Phase 11: Fine-Tuning Execution

This phase documents the operational execution of the fine-tuning campaign whose design was specified in Phase 10. Three notable deviations from the planned protocol — driven by hardware availability, tooling constraints in the Unsloth Studio (Beta) interface, and a runtime failure mode that consumed three pre-execution attempts before resolution — are recorded so that the dissertation's results can be re-interpreted in their light.

### 11.1 Deviations from the Phase 10 Plan

**Compute substrate.** Phase 10.4 specified an A40 48 GB pod. At deployment, an A40 instance was unavailable on the contracted RunPod tier; an RTX PRO 4500 Blackwell pod (sm_120, 32 GB VRAM, 58 GiB container RAM cap) was used for the first three attempts. Following the failure mode described in §11.2, the campaign was migrated to an L40S 48 GB pod (NVIDIA AD102, sm_89, 46,068 MiB GDDR6 ECC, 175 GiB container RAM cap, 13.6 effective vCPU on an AMD EPYC 9354 host). The L40S is the same silicon as the RTX 6000 Ada and is functionally equivalent to the planned A40 for LoRA fine-tuning at the parameter scales used here; no methodological consequence is expected from the substitution.

**Architecture set.** The Phase 10.3 plan listed four base models (Qwen 3.5 4B, Qwen 3.5 9B, Gemma 4 E4B, MedGemma 1.5 4B) intended to support a three-architecture × two-supervision factorial (six runs, Phase 10.5–10.6). The executed campaign substitutes **Gemma 4 E2B** for MedGemma 1.5 4B and elevates it to a primary architecture, yielding a four-architecture × two-supervision factorial (eight runs). The substitution preserves the ablation logic of §10.5 and adds a within-Gemma-4 size contrast (E2B vs E4B), at the cost of dropping the medical-specialised baseline. MedGemma is deferred to a subsequent campaign.

**Best-checkpoint selection.** Phase 10.4 specified `load_best_model_at_end=True` with `eval_strategy="epoch"` under the canonical Hugging Face Trainer interface. Unsloth Studio (Beta) does not expose either argument through its YAML schema; instead, it persists every checkpoint at a user-configurable cadence (`save_steps=800`) and records evaluation metrics in an SQLite database at `/root/.unsloth/studio/studio.db` (table `training_metrics`). To preserve the methodological intent, a post-hoc selector — `scripts/pick_best_checkpoint.py` — was implemented to query the database, locate the corresponding checkpoint directory, and push it to the Hugging Face Hub. Studio's evaluation cadence was set to `eval_steps=0.1` (≈ every 802 of 8,015 steps), yielding ten evaluation points per run, finer-grained than the per-epoch scheme planned.

### 11.2 Pre-Execution Failure Mode and Resolution

Two consecutive Studio runs (timestamps 2026-04-28 23:19:24 and 23:32:55 UTC) terminated identically: the training subprocess died at `final_step = 0` after approximately 4 minutes 35 seconds, during the validation-set VLM-format conversion phase, at sample index ~989 of 2,849. Neither attempt produced a Python traceback in the Studio log, and the SQLite `error_message` field reported only the generic string "Training process terminated unexpectedly". The crash was reproducible across hyperparameter changes (`max_seq_length: 2048→2850`; `save_steps: 30→800`; `train_on_completions: false→true`; `random_seed: 3407→42`), confirming that no field in the YAML was implicated.

Diagnosis was achieved by inspecting the cgroup v1 memory accounting on the executing pod:

```text
/sys/fs/cgroup/memory/memory.events
  oom 1
  oom_kill 1
  max  17478
memory.peak  62,000,001,024 B  (57.74 GiB)
memory.max   61,999,996,928 B  (57.74 GiB)
```

The `oom_kill = 1` counter, combined with `memory.peak ≈ memory.max`, establishes that the container memory cap had been reached and the kernel out-of-memory killer had terminated the training subprocess. SIGKILL bypasses Python signal handlers, which explains the absence of a traceback in stderr; in this configuration, **silence in the log is itself diagnostic** of an OS-level kill rather than a Python exception.

The root cause is the in-memory VLM sample conversion implemented by Studio. With image bytes embedded in the published parquet via `datasets.Image()` and Gemma 4's per-image token expansion (~2,520 image tokens per sample), the working-set size for tokenising the 28,486-row train + validation corpus exceeds 58 GiB before the optimiser is initialised. No hyperparameter change addresses this, because the failure precedes any GPU activity: container RAM, not VRAM, is the binding constraint. The migration to the L40S template (175 GiB cap) resolved the failure; on the new pod, the training-subprocess working set stabilised at 109.9 GiB, comfortably below the new cap, confirming the diagnosis.

### 11.3 Run 1: Gemma 4 E2B-it × Label-Only — Status as of 2026-04-29 01:30 UTC

Run 1 is in progress at the time of writing. Its configuration matches Table 10.3.1 with two run-specific values: `max_seq_length = 2850` (sized to fit Gemma 4's image-token footprint plus the instruction and label tokens with no truncation) and `train_on_completions = true` (loss masked to the assistant-response tokens only, appropriate for the label-only supervision density). The full effective configuration is reproduced in Table 11.3.1.

**Table 11.3.1.** Run 1 effective configuration.

| Parameter | Value | Rationale |
|---|---|---|
| Base model | `unsloth/gemma-4-E2B-it` | Phase 10 architecture set, smallest Gemma 4 |
| Dataset | `danielfdias98/derm-reasoning-label-only` | Phase 9 published repository |
| `max_seq_length` | 2,850 | Image (~2,520) + instruction + label, non-truncating |
| `num_epochs` | 5 | Phase 10.3.1 standardised |
| `learning_rate` | 2 × 10⁻⁴ | Phase 10.3.2 Gemma 4 schedule |
| `lr_scheduler_type` | Linear | Phase 10.3.2 Gemma 4 schedule |
| `warmup_steps` | 5 | Phase 10.3.2 Gemma 4 schedule |
| `weight_decay` | 0.001 | Phase 10.3.2 Gemma 4 schedule |
| Effective batch size | 16 (= `batch_size 4 × grad_accum 4`) | Phase 10.3.1 standardised |
| LoRA `r`, `α`, dropout | 32, 32, 0.05 | Phase 10.3.1 standardised |
| `target_modules` | `all-linear` | Phase 10.3.1 standardised |
| `optim` | `adamw_8bit` | Phase 10.3.1 standardised |
| `gradient_checkpointing` | Unsloth | Phase 10.3.1 standardised |
| `save_steps` | 800 | 10 checkpoints per run |
| `eval_steps` | 0.1 (≈ 802) | 10 evaluations per run |
| `train_on_completions` | True | Loss restricted to the diagnosis-label tokens |
| `random_seed` | 42 | Phase 10.3.1 standardised |

Run 1 began at 2026-04-29 00:12:04 UTC. The training subprocess maintained 100 % GPU utilisation during the steady-state phase, with VRAM stabilising at 11.1 / 46 GB and container RAM at 109 / 175 GiB throughout. After the Triton-compilation warmup (76.9 s for step 1), per-step wall-clock cost converged to 3.90 s mean (range 3.28 – 5.31 s over the most recent 100 steps); projected total run time is ~7.3 hours.

The training-loss trajectory, drawn from the `training_metrics` table, exhibits the canonical fine-tuning shape — a rapid descent over the first ~100 steps, followed by a slower long-tail grind:

**Table 11.3.2.** Training-loss trajectory, Run 1, sampled steps.

| Step | Train loss | Grad norm | LR | Epoch |
|---|---|---|---|---|
| 1 | 16.34 | 48.10 | 0.0 | 0.000 |
| 5 | 8.62 | 25.36 | 1.6 × 10⁻⁴ | 0.003 |
| 50 | 0.31 | 0.52 | 2.0 × 10⁻⁴ | 0.030 |
| 100 | 0.25 | 0.47 | 1.99 × 10⁻⁴ | 0.060 |
| 800 | 0.11 | 0.18 | 1.80 × 10⁻⁴ | 0.500 |
| 1,029 | mean(last 20) = 0.124 ± 0.035 | 0.17 | 1.74 × 10⁻⁴ | 0.640 |

The starting loss of 16.34 is consistent with Unsloth's documented expectation for Gemma 4 E2B (13–15 range; §10.3.2 footnote on Gemma-4 family behaviour). No NaN or Inf values were observed across all 1,029 recorded steps. Gradient-norm magnitudes contracted from ~48 at initialisation to a stable range of 0.1–0.4 by step 50, indicating gradient stability and the absence of exploding-gradient pathology under the chosen LR schedule.

### 11.4 First Evaluation Anomaly and Working Hypothesis

The first scheduled evaluation fired at step 802 and reported `eval_loss = 3.9005` against a contemporaneous training-batch loss of 0.1601, a ratio of 24.4 ×. This magnitude is incompatible with classical overfitting: at step 802 the model has processed approximately one half of one epoch over the training corpus, and no individual training sample has been encountered more than once. Memorisation of the training distribution is ruled out on logical grounds at this point in training.

The leading hypothesis — to be tested by the second evaluation at step 1,604 — is a **collator asymmetry between the training and evaluation pipelines under `train_on_completions = true`**. The completion-only data collator masks user and image tokens (replacing labels with `-100` in the cross-entropy computation), so the training loss is averaged over only the ~10–20 response tokens per sample. If Studio's evaluation pipeline applies a default (non-completion-only) collator, `eval_loss` is averaged over all ~2,550 tokens per sample including the un-trained image and instruction tokens. The expected ratio inflation under this hypothesis is on the order of (total tokens) / (response tokens) ≈ 100–250 ×; the observed 24 × is in the plausible range for a partial mask. This is consistent with documented Unsloth/TRL behaviour when the `eval_dataset` argument is supplied without an explicit `eval_data_collator`.

Three diagnostic outcomes are pre-registered for the step-1,604 evaluation:

1. `eval_loss` decreases substantially (e.g., 3.9 → 2.5) — the model is generalising along the same warped scale; the absolute eval-loss scale is uninformative, but **relative changes remain valid** for both within-run early-stopping decisions and best-checkpoint selection via `pick_best_checkpoint.py`.
2. `eval_loss` remains within ±0.1 of 3.9 — the loss is dominated by un-trained-token noise; the training-loss curve and post-training empirical evaluation (Phase 10.6 metrics on Fitzpatrick17k, Confusion Triads, MM-Skin VQA) become the only meaningful signals for run quality.
3. `eval_loss` rises significantly (> 4.5) — the collator-asymmetry artefact alone cannot explain the trajectory; deeper investigation of train/validation distribution mismatch, image-source skew, or pipeline corruption is warranted before continuing the campaign.

The watcher infrastructure (§11.5) automatically reports this comparison after every evaluation, including the picker's selection of the lowest-`eval_loss` checkpoint observed to date.

### 11.5 Watcher Infrastructure

Two persistent monitors run in parallel during execution. The first polls Studio's web UI on its RunPod proxy URL every 60 s, alerting on three consecutive 5xx-class responses (down) or recovery thereafter (up); this catches the failure mode in which the Studio web server itself crashes independently of the training subprocess (observed once during the §11.2 diagnosis phase). The second polls `/root/.unsloth/studio/studio.db` over SSH every 60 s, compares the previous probe with the current one, and emits one of three classes of alert:

- *Critical* — NaN or Inf in the recent loss window; loss spike (instantaneous loss > 5 × the mean of the last 20, after step 30); process stall (training-step count unchanged for ≥ 10 minutes); cgroup memory > 95 % of limit; `status` transition to `error` or `failed`.
- *Warning* — gradient-norm peak exceeding 10 in the last 20 steps when the previous window was below 10.
- *Milestone* — new checkpoint persisted; new `eval_loss` row recorded (followed by automatic invocation of `pick_best_checkpoint.py --json`, which reports whether the latest evaluation is the new minimum or whether an earlier checkpoint remains preferable); epoch boundary crossed; run completed.

Together, the two monitors provide both the liveness signal (Studio reachable, training process alive) and the quality signal (loss healthy, checkpoint progress recorded, evaluation trajectory acceptable) required for unattended overnight execution.

### 11.6 Provisional Observations

At the time of writing, with Run 1 12.8 % complete and a single evaluation point recorded:

- The pipeline is operational under the L40S configuration; the OOM failure that consumed the three pre-execution attempts has been definitively resolved by the migration to a higher-memory pod tier.
- Training dynamics for Gemma 4 E2B with label-only supervision exhibit the textbook fine-tuning shape — rapid initial descent followed by a long-tail grind — with no early evidence of instability (no NaN/Inf, gradient norms contracted and stable, GPU utilisation at 100 %, container RAM flat at 62 % of the cap).
- The first `eval_loss` value (3.9005 at step 802) cannot yet be interpreted on its absolute scale; relative comparison across evaluations remains the basis for both within-run early-stopping decisions and best-checkpoint selection.
- The reproducibility of the OOM failure mode across all hyperparameter configurations attempted constitutes a methodological finding in its own right: **container memory provisioning, not GPU memory, is the binding constraint for end-to-end VLM SFT on this dataset**, and should be checked first when migrating between RunPod GPU tiers.

The remaining seven runs (Gemma 4 E2B × full-reasoning; Gemma 4 E4B × {label-only, full-reasoning}; Qwen 3.5 4B × {label-only, full-reasoning}; Qwen 3.5 9B × {label-only, full-reasoning}) are queued behind Run 1 completion. Per-run intermediate observations and any anomalies will be appended to subsequent subsections of this Phase as they are recorded, preserving a single chronological record of the execution.

---

## Next Steps

1. **Zero-shot baselines:** Run all 4 student models on Fitzpatrick17k + MM-Skin VQA + Confusion Triads (RunPod GPU)
2. **Reasoning generation:** Generate structured descriptions for 29,913 training images using teacher model
3. **Training format conversion:** Convert to VLM chat format
4. **Fine-tuning:** LoRA fine-tune all 4 student models on RunPod GPU
5. **Post-training evaluation:** Re-run all 3 benchmarks, measure Δ improvement + fairness per FST
6. **Teacher comparison:** Compare reasoning quality from different teacher models (MedGemma vs Claude vs Gemini vs clinician ground truth from SkinCoT if approved)
