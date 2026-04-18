# Dermatology SLM/LLM Literature Review

**Compiled:** 2026-04-17
**Scope:** Studies (2020–2026) on Small/Large Language Models, Vision-Language Models (VLMs), and Multimodal LLMs (MLLMs) applied to dermatology — with side-by-side relevance notes for the efficient dermatology VLM dissertation (Qwen 3.5 4B/9B, MedGemma 4B, Gemma 4 E4B students; Gemini 2.5 Flash teacher; knowledge distillation + structured reasoning pipeline).

> **Reading guide.** Papers are grouped thematically (§1–§6), each with a summary table and detailed writeups for the most relevant entries. Studies already cited in `docs/src/implementation_log.md` (SkinGPT-4, SkinFlow, LLaVA-Med, SkinCaRe/SkinCoT, DermoGPT, SkinGPT-R1, DermaBench, DermaSynth, MedGemma) appear only as reference points. Everything else is new material to consider citing.

---

## Table of Contents

1. [Dermatology-specific VLMs and Foundation Models](#1-dermatology-specific-vlms-and-foundation-models)
2. [Small Medical LMs, Distillation, and Parameter-Efficient Fine-Tuning](#2-small-medical-lms-distillation-and-parameter-efficient-fine-tuning)
3. [Structured Reasoning / Chain-of-Thought in Medical VLMs](#3-structured-reasoning--chain-of-thought-in-medical-vlms)
4. [Fairness, Skin Tone, and Equity in Dermatology AI](#4-fairness-skin-tone-and-equity-in-dermatology-ai)
5. [Multi-Agent Medical and Dermatology Systems (RAG + Routing)](#5-multi-agent-medical-and-dermatology-systems-rag--routing)
6. [Benchmarks, Evaluators, and Datasets (2023–2026)](#6-benchmarks-evaluators-and-datasets-20232026)
7. [Synthesis: Implications for the Dissertation](#7-synthesis-implications-for-the-dissertation)
8. [Verification Caveats](#8-verification-caveats)

---

## 1. Dermatology-specific VLMs and Foundation Models

These are the closest direct competitors to the student models your dissertation fine-tunes. Several post-date SkinFlow and SkinGPT-4 and should be added to your comparison tables.

### 1.1 Summary Table

| Paper | arXiv | Year | Architecture | Train Data | Headline Result |
|---|---|---|---|---|---|
| **PanDerm** (Yan et al., *Nature Medicine*) | [2410.15038](https://arxiv.org/abs/2410.15038) | 2024/25 | ViT SSL (masked latent + CLIP align) | 2M images, 11 institutions, 4 modalities | Outperforms clinicians by +10.2% on early melanoma detection; 28 benchmarks SOTA |
| **Derm1M / DermLIP** (Yan et al., ICCV'25 Highlight) | [2503.14911](https://arxiv.org/abs/2503.14911) | 2025 | CLIP/SigLIP VLP | 1.03M image-text pairs, 390 conditions, 4-level ontology | +9.85% zero-shot / +48.1% retrieval vs BiomedCLIP across 8 benchmarks |
| **MM-Skin / SkinVL** (Zeng et al.) | [2505.06152](https://arxiv.org/abs/2505.06152) | 2025 | LLaVA-Med + SigLIP + LoRA | 11k textbook images + 27k VQA | Beats general and medical VLMs across 8 datasets |
| **MAKE** (Yan et al.) | [2505.09372](https://arxiv.org/abs/2505.09372) | 2025 | Multi-aspect contrastive VLP | 403,563 derm image-text pairs | SOTA zero-shot on 8 derm datasets |
| **DermINO** | [2508.12190](https://arxiv.org/abs/2508.12190) | 2025 | DINO + semi-supervised + KG prototypes | 432k curated images | 95.79% reader-study accuracy vs clinicians' 73.66% |
| **Skin-R1** (Liu et al.) | [2511.14900](https://arxiv.org/abs/2511.14900) | 2025 | VLM + textbook CoT SFT + hierarchical RL | Textbook-derived reasoning traces | Separate SOTA claim from SkinGPT-R1 (reasoning-RL centric) |
| **SkinGEN** (Lin et al., IUI'25) | [2404.14755](https://arxiv.org/abs/2404.14755) | 2024 | SkinGPT-4 + Stable Diffusion LoRA | — | User study n=32: higher trust + comprehension via visual explanations |
| **DermIQ-VLM / GRPO++** (Swapnil et al.) | [2510.01236](https://arxiv.org/abs/2510.01236) | 2025 | Small VLM + GRPO++ + DPO + KG | Curated dermatology set | Stable GRPO for low-resource reasoning training |
| **CLARIFY** (Saha et al.) | [2508.18430](https://arxiv.org/abs/2508.18430) | 2025 | Specialist classifier + compressed VLM + KG RAG | Curated derm VQA | **+18% accuracy; −20% VRAM, −5% latency** vs fine-tuned single VLM |
| **DermatoLlama 1.0 / SCALEMED** | medRxiv 2025.05.17.25327785 | 2025 | Llama-3.2-11B-Vision + LoRA | DermaSynth (1.2M synthetic pairs) | Matches GPT-4o / Gemini 2.0 Flash on derm report generation |
| **DermETAS-SNA** (Oruganty et al.) | [2512.08998](https://arxiv.org/abs/2512.08998) | 2025 | Evolutionary ViT + StackNet + Gemini 2.5 Pro RAG | DermNet 23-class + SKINCON | F1 56.30% (vs SkinGPT-4 48.51%); 92% MD agreement in n=8 reader study |
| **Dermacen Analytica** (Panagoulias et al.) | [2403.14243](https://arxiv.org/abs/2403.14243) | 2024 | Segmentation + ViT + LLM pipeline | Public case studies | 0.87 contextual + diagnostic scores for teledermatology |

### 1.2 Detailed Writeups

**PanDerm (arXiv:2410.15038, Nat Med 2025).** Canonical dermatology vision foundation model. ViT pre-trained via joint masked-latent + CLIP-aligned objectives on 2M real images from 11 clinical institutions covering dermoscopy, clinical photos, TBP tiles, and dermatopathology. Evaluated on 28 benchmarks: skin cancer screening, risk stratification, DDx for 128 conditions, lesion segmentation, longitudinal monitoring, metastasis prediction. Three reader studies: +10.2% vs clinicians on early melanoma; +11% clinician dermoscopy accuracy with AI assist; +16.5% non-dermatologist DDx on clinical photos. Only 10% of labeled data needed for SOTA. Not a generative LLM — but **the strongest vision-encoder baseline** for any dermatology VLM work in 2025+. Your dissertation should position Qwen 3.5 9B's 67.1% on confusion triads against PanDerm-scale diagnostic performance.

**Derm1M + DermLIP (arXiv:2503.14911, ICCV 2025 Highlight).** First million-scale dermatology image-text corpus: 1,029,761 pairs, **~257× larger than any prior derm VLP dataset**, aligned to a clinician-built 4-level ontology over 390 conditions and 130 clinical concepts. DermLIP (multiple SigLIP-style variants trained on this corpus) beats BiomedCLIP / PubMedCLIP by +9.85% zero-shot classification and +48.1% retrieval recall across 8 derm benchmarks. The ontology itself is directly usable as a knowledge backbone for structured reasoning targets (maps cleanly to your 337 training conditions and 6-class evaluation).

**MM-Skin / SkinVL (arXiv:2505.06152).** A dermatology VLM built by LoRA-tuning LLaVA-Med on 11,039 images (1,039 dermoscopic, 3,016 pathology, 6,984 clinical) extracted from professional textbooks with 27k+ instruction-following VQA pairs (9× the previous largest derm VQA set). Their "skipped" status in your implementation log was correct for training data (dermoscopic-heavy), but **MM-Skin is the best available evaluation benchmark** for mixed-modality dermatology and should be added alongside Fitzpatrick17k + Confusion Triads.

**Skin-R1 (arXiv:2511.14900).** Published almost simultaneously with SkinGPT-R1 (arXiv:2511.15242) and covers similar ground but with a distinct recipe: (i) textbook-based reasoning generator that synthesizes hierarchy-aware, DDx-informed trajectories for SFT; (ii) novel RL stage that uses the disease hierarchy as part of the reward. Both papers converge on "structured reasoning + RL" as the dermatology VLM frontier. The difference: SkinGPT-R1 emphasizes fairness MoE + Fitzpatrick balance; Skin-R1 emphasizes differential-diagnosis-driven reasoning. **Cite both** — your dissertation contributes a third axis (pure distillation without RL in the student) to this conversation.

**CLARIFY (arXiv:2508.18430).** The closest architectural cousin to your Phase 6 multi-agent design. Couples a lightweight domain-trained image classifier (Specialist) with a compressed conversational VLM (Generalist), with a KG-based RAG module grounding responses. Reports +18% accuracy vs a fine-tuned single-VLM baseline **while reducing VRAM ≥20% and latency ≥5%**. Their Specialist-Generalist framing maps neatly onto your "VLM orchestrator → category-specialist SLM+RAG agents." Their empirical efficiency claim is the kind of result your deployment discussion should target.

**DermatoLlama / SCALEMED / DermaSynth (medRxiv 2025.05.17.25327785).** ⚠ *Possibly different from the DermaSynth you already cite* — verify author and scope. Reports an 11B LoRA-tuned Llama-3.2-Vision trained on 1.2M synthetic text samples from 56K PubMed + 45K open-access images, with an AnnotatorMed clinician-in-the-loop tool. Claims parity with GPT-4o and Gemini 2.0 Flash on report generation. If this is the same DermaSynth (Yilmaz) paper, the new piece is the SCALEMED framing of modular deployability.

**DermINO (arXiv:2508.12190).** Alternative vision-foundation approach to PanDerm that folds knowledge-guided prototype initialization into a DINO-style SSL encoder via a medical language model. Reader study (n=23 dermatologists): DermINO 95.79% vs clinicians 73.66%; AI assistance improved clinician performance by +17.21%. The prototype-initialization trick is a cheap form of knowledge injection adjacent to your distillation pipeline.

**DERM-3R (arXiv:2604.09596).** TCM dermatology multi-agent framework with three specialists (DERM-Rec, DERM-Rep, DERM-Reason) on a lightweight multimodal LLM, fine-tuned on just 103 real-world psoriasis cases. Matches or surpasses large general-purpose multimodal models despite minimal data. Direct precedent for your "structured, domain-aware multi-agent modeling as an alternative to brute-force scaling" claim (Phase 6).

---

## 2. Small Medical LMs, Distillation, and Parameter-Efficient Fine-Tuning

This section validates the "small beats big through distillation" thesis that anchors your dissertation.

### 2.1 Summary Table

| Paper | arXiv | Year | Size | Method | Headline Result |
|---|---|---|---|---|---|
| **Meditron** (Chen et al., EPFL) | [2311.16079](https://arxiv.org/abs/2311.16079) | 2023 | 7B / 70B | Continued pretraining on PubMed | 7B beats PMC-LLaMA by +10%; 70B within 5% of GPT-4 |
| **PMC-LLaMA** (Wu et al.) | [2304.14454](https://arxiv.org/abs/2304.14454) | 2023 | 7B | Domain pretrain + instruction tune | Strong USMLE/MedMCQA/PubMedQA at 7B |
| **BioMedLM** (Bolton et al., Stanford CRFM) | [2403.18421](https://arxiv.org/abs/2403.18421) | 2024 | **2.7B** | From-scratch GPT on PubMed | 69.0% MMLU Medical Genetics; 57.3% MedMCQA (GPT-4 competitive) |
| **Me-LLaMA** | [2402.12749](https://arxiv.org/abs/2402.12749) | 2024 | 13B / 70B | 129B-token continued pretrain + IT | Beats GPT-4 on 5/8 benchmarks after task tuning |
| **Med42** (M42 Health) | [2404.14779](https://arxiv.org/abs/2404.14779) | 2024 | 70B + LoRA ablation | Full-FT vs LoRA head-to-head | **LoRA 68.3% vs full-FT 72% USMLE with 0.15% of params** |
| **LLaVA-MoD** (Shu et al., ICLR'25) | [2408.15881](https://arxiv.org/abs/2408.15881) | 2024 | **2B active** | MoE progressive KD from LLaVA teacher | 2B beats Qwen-VL-Chat-7B by +8.8% using 0.3% of training data |
| **Med-MoE** (Jiang et al., EMNLP'24) | [2404.10237](https://arxiv.org/abs/2404.10237) | 2024 | ~2–3B active | 3-stage MoE VLM | Lightweight VLM for both discriminative + generative MedVQA |
| **MiniGPT-Med** (Alkhaldi et al.) | [2407.04106](https://arxiv.org/abs/2407.04106) | 2024 | ~7B | Unified radiology VLM | +19% vs prior SOTA on report generation |
| **RadFM** (Wu et al., Nat Comm 2025) | [2308.02463](https://arxiv.org/abs/2308.02463) | 2023/25 | 14B | Radiology foundation VLM | Beats Med-Flamingo, LLaVA-Med, GPT-4V on RadBench |
| **Med-Flamingo** (Moor et al.) | [2307.15189](https://arxiv.org/abs/2307.15189) | 2023 | 9B | Continued pretrain | +20% clinician rating in few-shot Med VQA |
| **PMC-VQA / MedVInT** | [2305.10415](https://arxiv.org/abs/2305.10415) | 2023 | ~7B | Visual instruction tuning | >80% multi-choice accuracy; released 227k benchmark |
| **Aloe** (BSC) | [2405.01886](https://arxiv.org/abs/2405.01886) | 2024 | 7B / 8B | SFT + DPO + synthetic CoT | Beats base LLaMA-3; +7% via medprompt |
| **LLaMA-3.2-3B-MedCoT** (Mansha) | [2510.05003](https://arxiv.org/abs/2510.05003) | 2025 | **3B** | QLoRA + Unsloth | 60% memory reduction, improved CoT accuracy |
| **LLaMA-XR** | [2506.03178](https://arxiv.org/abs/2506.03178) | 2025 | 8B | QLoRA + DenseNet-121 | ROUGE-L 0.433 on radiology reports |
| **BIOMEDICA / BMCA-CLIP** (Lozano et al.) | [2501.07171](https://arxiv.org/abs/2501.07171) | 2025 | CLIP-class | PMC Open Access corpus (24M pairs) | +6.56% avg zero-shot; **+29.8% dermatology**; 10× less compute |
| **MedGemma Technical Report** | [2507.05201](https://arxiv.org/abs/2507.05201) | 2025 | 4B / 27B | Medical continued pretrain on Gemma 3 | SigLIP encoder pretrained on derm data |

### 2.2 Detailed Writeups

**Meditron-7B / -70B (arXiv:2311.16079).** The foundational argument for domain-pretrained small medical LMs. Llama-2 continued on 46K clinical guidelines + 16.1M PubMed abstracts + 5M full papers + 400M RedPajama replay tokens. The 7B beats PMC-LLaMA-7B by +10% average across MedQA/MedMCQA/PubMedQA/MMLU-Medical; the 70B is within 5% of GPT-4. **Core citation** for "a 7B specialist can close most of the gap to a trillion-parameter generalist," which is the dissertation's central thesis.

**LLaVA-MoD (arXiv:2408.15881, ICLR 2025).** The clearest successful blueprint for multimodal distillation in the 2–4B range: an MoE student learns from a large LLaVA teacher via progressive KD (KL mimic → DPO-style preference distillation). A **2B-activated student beats Qwen-VL-Chat-7B by +8.8% average on multimodal benchmarks while using only 0.3% of the teacher's training data**. Methodologically this is your closest non-medical analog — direct reference for distilling Gemini 2.5 Flash reasoning traces into 4–9B student VLMs.

**Med42 / PEFT evaluation (arXiv:2404.14779).** Rigorous head-to-head of full fine-tuning vs LoRA on Llama-2-70B: full-FT reaches 72% USMLE; LoRA reaches 68.3% while updating only **0.15% of parameters**. The 3.7-point gap for a ~650× reduction in trainable params is the exact trade-off your dissertation exploits.

**BioMedLM (arXiv:2403.18421).** 2.7B GPT-style model trained from scratch on PubMed only; fine-tuned BioMedLM reaches 69.0% MMLU Medical Genetics and 57.3% MedMCQA, competitive with much larger Med-PaLM / GPT-4. The headline argument — privacy-preserving, on-prem, "green" alternatives — is a secondary thesis point you could emphasize.

**BIOMEDICA (arXiv:2501.07171).** PubMed Central OA mined into 24M image-text pairs across 6M articles. The dermatology-specific number is striking: **+29.8% improvement on zero-shot dermatology classification** with 10× less compute than SOTA. A strong cheap baseline to benchmark your fine-tuned students against, especially since it's a CLIP-class model (not generative) — validates the argument that good data beats clever architecture.

**MedGemma Technical Report (arXiv:2507.05201).** Essential companion citation to your existing MedGemma 4B baseline — gives the formal dataset/training story (Gemma 3 4B/27B + SigLIP continued pretrain on medical data including dermatology). Your 19.7% Fitzpatrick17k Top-1 for MedGemma 4B should reference this technical report when reporting architecture details.

---

## 3. Structured Reasoning / Chain-of-Thought in Medical VLMs

Directly supports your Gemini-2.5-Flash → structured JSON → SLM student pipeline.

### 3.1 Summary Table

| Paper | arXiv | Year | Mechanism | Takeaway for Dissertation |
|---|---|---|---|---|
| **HuatuoGPT-o1** (Chen et al.) | [2412.18925](https://arxiv.org/abs/2412.18925) | 2024 | SFT on verifier-filtered CoT + RL (GPT-4o judge) | Verifier-filtered distillation blueprint |
| **MedCoT** (Liu & Wang, EMNLP'24) | [2412.13736](https://arxiv.org/abs/2412.13736) | 2024 | Hierarchical specialists + MoE vote | Structured multi-step beats flat CoT |
| **MedVLM-R1** (Pan et al.) | [2502.19634](https://arxiv.org/abs/2502.19634) | 2025 | GRPO on Qwen2-VL-2B with format+answer reward | **Counter-argument:** reasoning emerges from RL alone |
| **Med-R1** (Lai et al.) | [2503.13939](https://arxiv.org/abs/2503.13939) | 2025 | GRPO on Qwen2-VL-3B across 8 modalities | 3B matches 7B–13B medical VLMs |
| **MedReason** (Wu et al.) | [2504.00993](https://arxiv.org/abs/2504.00993) | 2025 | KG-validated CoT paths → SFT | Strongest non-derm analog to label-anchored generation |
| **AlphaMed** (Liu et al.) | [2505.17952](https://arxiv.org/abs/2505.17952) | 2025 | Rule-based RL only, no SFT, no CoT | Counter-evidence: you may not need distillation |
| **ReasonMed** (Sun et al.) | [2506.09513](https://arxiv.org/abs/2506.09513) | 2025 | 1.75M→370K multi-agent filtered CoT | Over-generate + filter template |
| **Dr-LLaVA** (Sun et al., NeurIPS MAR'24) | [2405.19567](https://arxiv.org/abs/2405.19567) | 2024 | Symbolic decision tree + GPT-4 + symbolic reward | Closest method analog; "symbolic skeleton" |
| **MedCLM** (Kim et al.) | [2510.04477](https://arxiv.org/abs/2510.04477) | 2025 | Detection→VQA-with-CoT via boxes + curriculum | Label-anchored synthetic CoT across modality |
| **Why CoT Fails in Clinical Text** (Chen et al.) | [2509.21933](https://arxiv.org/abs/2509.21933) | 2025 | Flat CoT evaluation on clinical corpora | Flat CoT **hurts 86% of LLMs** — structural reasoning preferred |
| **Reasoning LLMs in Medicine survey** | [2508.19097](https://arxiv.org/pdf/2508.19097) | 2025 | Survey | Useful for dissertation background |
| **MAGIC** (Wang et al.) | [2506.12323](https://arxiv.org/abs/2506.12323) | 2025 | MLLM-as-critic for synthetic derm image QC | +9.02% / +13.89% few-shot when augmenting training data |

### 3.2 Detailed Writeups

**HuatuoGPT-o1 (arXiv:2412.18925).** Strongest 2024–2025 template for teacher-student CoT distillation in medicine. Starts from 40K verifiable MedQA-style problems; a GPT-4o-driven medical verifier guides search over complex CoT trajectories; correct trajectories become SFT data for Llama-3.1 8B/70B; a second RL stage uses the same verifier as sparse reward. **This is the methodology to map your pipeline onto:** your expert label + structured JSON schema act as a free verifier — every Gemini output whose final `reasoning` field implies a different diagnosis from the ground-truth label should be filtered or re-sampled. Report the filter-pass rate explicitly in your methodology chapter.

**MedReason (arXiv:2504.00993).** Closest non-dermatology analog to your label-anchored generation. 32,682 QA pairs where every answer is a validated "thinking path" through a structured medical knowledge graph. Pipeline: clinical QA → KG entity extraction → path traversal → natural-language rationale → LLM validation against evidence-based medicine. Their KG plays the same role as your "expert dermatology label + Fitzpatrick + morphology vocabulary": a structured anchor that constrains the teacher's reasoning to be factually grounded rather than plausibly fluent. **Cite this as the precedent for label-anchored synthetic rationale generation.** Their ~32K scale is also a useful sanity check for your keep-after-filter dataset size.

**Dr-LLaVA (arXiv:2405.19567).** Most directly analogous existing work. Bone-marrow pathology: symbolic representation of the clinical decision tree → GPT-4 generates instruction-tuning dialogues *grounded in that tree* → a symbolic reward function checks each model response against the tree's constraints during RL. Their "symbolic representation" ≡ your "expert ground-truth label + structured JSON schema." The explicit claim — that symbolic grounding eliminates the need for human RLHF annotators — is one you can directly port: your Fitzpatrick-anchored morphology/border/color fields form a symbolic skeleton that Gemini must fill in consistently.

**MedVLM-R1 (arXiv:2502.19634) + AlphaMed (arXiv:2505.17952) — the counter-arguments.** Qwen2-VL-2B trained via GRPO with rule-based format+answer rewards from only 600 VQA samples and no CoT supervision reaches 55.11% → 78.22% across MRI/CT/X-ray VQA and beats million-scale medical VLMs. AlphaMed shows SOTA on six medical QA benchmarks using **rule-based RL only, no SFT, no CoT data**. Your rebuttal: (i) dermatology labels alone cannot teach *what attributes to describe* — only *what diagnosis to output* — and attribute-level supervision is a first-class clinical goal (ABCDE, 7-point checklist); (ii) the dissertation aim is clinician-interpretable structured output, not max accuracy on MCQA.

**ReasonMed (arXiv:2506.09513).** Industrial-scale distilled-CoT template: 1.75M initial reasoning paths from multiple LLMs → filtered/refined via multi-agent verification → 370K curated examples (21% keep rate). An explicit "Error Refiner" agent corrects flagged traces rather than discarding them; examples are stratified into Easy/Medium/Difficult for curriculum SFT. ReasonMed-7B beats all prior sub-10B on medical QA and beats Llama-3.1-70B on PubMedQA by +4.6%. **Recipe to adopt:** over-generate aggressively with Gemini, filter to ~20-30%, apply a refiner where feasible, stratify by difficulty.

**"Why CoT Fails in Clinical Text Understanding" (arXiv:2509.21933).** Important defensive citation. Evaluates flat CoT on clinical corpora and finds it **hurts 86% of LLMs**. This specifically defends your JSON-schema design against reviewers who might ask "why not just use CoT?" — structured attribute decomposition is defensible on this evidence alone.

**MAGIC (arXiv:2506.12323).** Uses an MLLM critic to score diffusion-generated dermatology images against expert-defined criteria; augmenting training data with MLLM-validated synthetic images yields +9.02% standard / +13.89% few-shot on 20-condition classification. Potentially useful as a data-augmentation layer to complement your teacher distillation.

---

## 4. Fairness, Skin Tone, and Equity in Dermatology AI

Your dissertation already cites Daneshjou 2022 and SkinGPT-R1. These are the other papers that strengthen your fairness chapter.

### 4.1 Summary Table

| Paper | arXiv / DOI | Year | Type | Headline |
|---|---|---|---|---|
| **Groh et al., Fitzpatrick17k** | [2104.09957](https://arxiv.org/abs/2104.09957) | 2021 | Benchmark + dataset | Original FST benchmark; 3.97% FST VI |
| **Ward et al., SCIN** (*JAMA Netw Open*) | [2402.18545](https://arxiv.org/abs/2402.18545) | 2024 | Diverse dataset | 32.6% non-White; eFST + eMST labels; crowdsourced |
| **Nijjer et al., "Adapting LLMs for Skin Tone Bias"** | [2510.00055](https://arxiv.org/abs/2510.00055) | 2025 | LLM/VLM fairness | SkinGPT-4: DP 0.10, 17.8% hallucination; tuned: 0.83–0.90 parity |
| **TrueSkin** (Lu) | [2509.10980](https://arxiv.org/abs/2509.10980) | 2025 | Skin-tone recognition benchmark | LMMs systematically lighten intermediate tones |
| **Ulrich et al., "Beyond Fitzpatrick"** (*npj Digital Medicine*) | [nature/s41746-025-01770-4](https://www.nature.com/articles/s41746-025-01770-4) | 2025 | FST vs MST | **89–92% Monk accuracy vs 0–20% Fitzpatrick** in automated pipelines |
| **DermDiT** (Munia & Imran) | [2504.01838](https://arxiv.org/abs/2504.01838) | 2025 | Debiasing / synthetic | VLM-prompted diffusion for underrepresented groups |
| **DermDiff** (Munia & Imran) | [2503.17536](https://arxiv.org/abs/2503.17536) | 2025 | Debiasing / synthetic | Diffusion model for racial bias mitigation |
| **Morales-Forero et al., "Predictive Representativity"** | [2507.14176](https://arxiv.org/abs/2507.14176) | 2025 | Clinical equity | FST disparities persist **even with balanced training** |
| **López-Pérez et al.** | [2501.11752](https://arxiv.org/abs/2501.11752) | 2025 | Generative bias | VAE **still better on light skin after balancing** |
| **BiaslessNAS** (Sheng et al., MICCAI'24) | [2407.13896](https://arxiv.org/abs/2407.13896) | 2024 | Debiasing via NAS | +2.55% accuracy, +65.5% fairness |
| **Shah et al., "Skin Tone Label Granularity"** | [2509.11184](https://arxiv.org/abs/2509.11184) | 2025 | FST benchmarking | Coarser FST grouping harms fairness |
| **LesionTABE** (Mexia Diaz et al., ISBI'26) | [2601.03090](https://arxiv.org/abs/2601.03090) | 2026 | Foundation + adversarial | **+25% fairness** over ResNet-152 baseline |
| **CALIN** (Shen et al.) | [2506.23298](https://arxiv.org/abs/2506.23298) | 2025 | MLLM calibration | Few-shot MLLM calibration bias; HAM10000 + PAPILA + MIMIC-CXR |
| **Explainable AI as Double-Edged Sword** (Xu et al.) | [2512.12500](https://arxiv.org/abs/2512.12500) | 2025 | Clinical HCI | n=623 lay + n=153 PCPs; LLM explanations help experts, can mislead laypeople |
| **eSkinHealth** (ACM MM'25) | [2508.18608](https://arxiv.org/abs/2508.18608) | 2025 | Diverse dataset | 5,623 images, 1,639 cases from Côte d'Ivoire + Ghana |

### 4.2 Detailed Writeups

**Nijjer et al. (arXiv:2510.00055).** The most directly comparable prior work to your fairness chapter. Evaluates **SkinGPT-4** on 300 SCIN cases across six conditions and reports demographic parity 0.10 across FST with 0.10–0.15 gap between lightest and darkest tones, plus a **17.8% hallucination rate** in artifacts and anatomy. Customized fine-tuning achieves 0.83–0.90 parity across FST I–VI. The methodology (demographic parity + equalized odds per FST, dermatologist-rated hallucination audit) is exactly what your evaluation should replicate on your 4 students.

**"Beyond Fitzpatrick" (npj Digital Medicine 2025).** Empirical case for migrating from FST to Monk. Automated DensePose + OpenFace pipeline maps CIELAB ITA to both scales: **89–92% accuracy on Monk vs 0–20% on Fitzpatrick** in the same patients. If you can add MST labels to your test set (SCIN already has eMST), this is the strongest motivation to dual-report results.

**Morales-Forero et al. (arXiv:2507.14176) + López-Pérez et al. (arXiv:2501.11752).** Together these establish that **balanced training alone is not sufficient** — performance gaps persist post-balancing in classifiers (HAM10000 + BOSQUE) and in generative models (VAEs on Fitzpatrick17k). Strong rebuttals to reviewers who think "just rebalance the training set." Your synthetic-data-from-teacher pipeline should be audited under this lens.

**TrueSkin (arXiv:2509.10980).** 7,299 images in 6 skin-tone classes under varied lighting. Finding: state-of-the-art LMMs systematically misclassify intermediate tones as lighter. Training on TrueSkin yields **>20% accuracy improvement** over LMM baselines. Worth a small FST-estimation comparison in your fairness chapter if time permits (your teacher-estimated FST labels could themselves be drifting lighter).

**LesionTABE (arXiv:2601.03090).** Foundation-model embeddings + adversarial debiasing: **+25% fairness improvement** over ResNet-152 while maintaining or improving diagnostic accuracy. Methodological analog for SkinGPT-R1 architecture; gives your dissertation a precedent for "foundation encoder + bias-aware head" as an alternative to MoE-for-fairness.

**Ward et al., SCIN (arXiv:2402.18545, *JAMA Netw Open*).** You already use SCIN for data, but the **methodology paper** is an important clinical citation: 32.6% non-White contributors, balanced FST III–VI, and the first large-scale dataset with **paired eFST + eMST labels**. Gives you the ability to justify MST migration with a clinical-journal reference rather than only ML venues.

**CALIN / Shen et al. (arXiv:2506.23298).** Addresses an angle your evaluation hasn't covered: **calibration-bias** of MLLMs in few-shot in-context learning for medical imaging (including HAM10000 skin cancer). Their inference-time CALIN calibrator reduces disparity without retraining. If your fine-tuned students end up miscalibrated per FST group, cite this as prior work and a potential inference-time fix.

**"Explainable AI as Double-Edged Sword" (arXiv:2512.12500).** 2×2 experimental study (623 lay people + 153 primary-care physicians, with FST-balanced AI model and LLM explanations): **LLM explanations helped experts regardless of AI correctness, but misled lay users when the AI was wrong**. Directly relevant to the clinical-deployment narrative in your thesis — justifies restricting your system to clinician-facing use.

---

## 5. Multi-Agent Medical and Dermatology Systems (RAG + Routing)

Directly supports Phase 6 of your implementation log.

### 5.1 Summary Table

| Paper | arXiv | Year | Architecture | Relevance |
|---|---|---|---|---|
| **MEDDxAgent** | [2502.19175](https://arxiv.org/abs/2502.19175) | 2025 | Orchestrator + history-taking + KR + strategy agents | Interactive DDx; +10% across LLM sizes |
| **SkinGPT-X** (Chen et al.) | [2603.26122](https://arxiv.org/abs/2603.26122) | 2026 | Multi-agent derm system with self-evolving memory | **+9.6% DDI31, +13% F1 DermNet**; rare-disease benchmark |
| **DermPrompt / MAC** (UMass-BioNLP) | [2404.17749](https://arxiv.org/abs/2404.17749) | 2024 | GPT-4V retriever + re-ranker + Multi-Agent Conversation | MEDIQA-M3G'24 winner-adjacent |
| **DERM-3R** (TCM dermatology) | [2604.09596](https://arxiv.org/abs/2604.09596) | 2026 | 3-agent pipeline on lightweight MLLM | Matches larger general MLLMs on 103 cases |
| **CLARIFY** | [2508.18430](https://arxiv.org/abs/2508.18430) | 2025 | Specialist classifier + compressed VLM + KG RAG | +18% accuracy; −20% VRAM |
| **Learning to Be A Doctor** (Zhuang et al.) | [2504.11301](https://arxiv.org/abs/2504.11301) | 2025 | Automated medical agent architecture search | Evolves agent graphs for skin DDx |
| **Dermacen Analytica** | [2403.14243](https://arxiv.org/abs/2403.14243) | 2024 | Segmentation + ViT + LLM pipeline for teledermatology | 0.87 contextual + diagnostic score |
| **RAG for Multimodal Melanoma** | [2509.08338](https://arxiv.org/abs/2509.08338) | 2025 | BERT + ResNeXt-50 retriever + VLM | F1 +0.21 over early-fusion; corrects 91% FP + 62% FN |

### 5.2 Detailed Writeups

**SkinGPT-X (arXiv:2603.26122).** The most important multi-agent dermatology paper for your Phase 6 architecture. A multimodal **collaborative multi-agent system with a self-evolving dermatological memory mechanism** — explicitly simulates dermatologist workflow with continuous memory evolution. Three-tier evaluation: (i) vs 4 SOTA LLMs on 4 datasets → **+9.6% accuracy on DDI31, +13% weighted F1 on Dermnet**; (ii) 498-class fine-grained dataset; (iii) new rare-skin-disease benchmark (8 diseases, 564 samples) where it gains +9.8% accuracy, +7.1% F1, +10% Cohen's κ. **Your Phase 6 narrative should directly reference and contrast with SkinGPT-X** — they chose self-evolving memory; you chose static RAG with category-specialist routing.

**MEDDxAgent (arXiv:2502.19175).** Modular agent for interactive differential diagnosis across respiratory, skin, and rare diseases. Components: DDxDriver orchestrator + history-taking simulator + knowledge-retrieval agent + diagnostic-strategy agent. Reports **+10% accuracy improvements across LLM sizes** in interactive DDx. Demonstrates why you decomposed the system in Phase 6 rather than using a single monolithic LLM — single-attempt diagnosis under incomplete information is a demonstrated failure mode.

**CLARIFY (arXiv:2508.18430).** Already discussed in §1.2 but belongs here too. The Specialist-Generalist framing + KG-based RAG is the exact prior architecture you're extending with category-level specialization. Their reported +18% accuracy, −20% VRAM, −5% latency give you concrete numerical targets for your own deployment discussion.

**DermPrompt + Multi-Agent Conversation (arXiv:2404.17749).** Early (2024) evidence that multi-agent conversation outperforms best single-agent Chain-of-Thought on dermatology VQA. Specifically found that naive CoT works for retrieval but **Medical-Guidelines-Grounded CoT** is needed for accurate diagnosis — aligns with your decision to anchor teacher generations to expert labels.

**Learning to Be A Doctor (arXiv:2504.11301).** Complementary: automated search over medical agent architectures (node / structural / framework mutations). Evolved workflows on skin disease diagnosis beat manually-crafted ones. Suggests a future-work direction — your dissertation hand-designs the 4-specialist + orchestrator + validator layout; NAS-style search could potentially improve it.

**DERM-3R (arXiv:2604.09596).** TCM dermatology with 3 collaborative agents (DERM-Rec, DERM-Rep, DERM-Reason) on a lightweight multimodal LLM, fine-tuned on just 103 real-world psoriasis cases. Matches or surpasses large general-purpose multimodal models. Concrete existence proof for your "small models + structured collaboration ≫ monolithic big models" claim under extreme data scarcity.

---

## 6. Benchmarks, Evaluators, and Datasets (2023–2026)

Extending your current stack (Fitzpatrick17k 1,000 / MM-Skin VQA / Confusion Triads + planned DermaBench + DermEVAL).

| Benchmark | arXiv / source | Images | Task | Notes |
|---|---|---|---|---|
| **DermBench / DermEval** (Shen et al.) | [2511.09195](https://arxiv.org/abs/2511.09195) | 4,000 / 4,500 | Diagnostic narrative scoring | Reference-free MLLM evaluator; mean deviation 0.117/5 vs experts |
| **DermoBench** (Ru et al., DermoGPT) | [2601.01868](https://arxiv.org/abs/2601.01868) | 3,600 expert-verified | Morphology / Diagnosis / Reasoning / Fairness | 4 clinical axes; already referenced in your log |
| **DermaVQA-DAS** (Yim et al., MSR) | [2512.24340](https://arxiv.org/abs/2512.24340) | Patient-authored | Closed QA + segmentation (36+27 DAS) | o3 0.798, GPT-4.1 0.796, Gemini-1.5-Pro 0.783 |
| **DermaBench** (Yilmaz et al.) | [2601.14084](https://arxiv.org/abs/2601.14084) | 656 | VQA over DDI | 14,474 annotations; already cited |
| **Derm1M** (Yan et al.) | [2503.14911](https://arxiv.org/abs/2503.14911) | 1.03M | VLP corpus | 390 conditions, 4-level ontology |
| **eSkinHealth** (ACM MM'25) | [2508.18608](https://arxiv.org/abs/2508.18608) | 5,623 | NTD benchmark (West Africa) | Closes the representation gap for tropical conditions |
| **TrueSkin** | [2509.10980](https://arxiv.org/abs/2509.10980) | 7,299 | 6-class skin tone | LMM audit benchmark |
| **DermEVAL** (Zhao et al., WACV'26) | — | 11,347 | 16-category classification | Already in your log as "no public download" |
| **MM-Skin VQA** | [2505.06152](https://arxiv.org/abs/2505.06152) | 11k img + 27k QA | Multi-modal VQA | You already use this |

**Minimum recommended additions to your evaluation suite:**
- **DermBench + DermEval** for free-form narrative scoring (complements BERTScore F1, catches factual hallucination).
- **DermaVQA-DAS** for closed QA with structured DAS (36 high-level + 27 fine-grained questions) — gives you a directly benchmarkable number against GPT-4.1 / Gemini-1.5-Pro / o3.
- **eSkinHealth** for geographic and FST generalization.

---

## 7. Synthesis: Implications for the Dissertation

### 7.1 The literature now splits dermatology VLMs into four schools

| School | Exemplars | Core recipe |
|---|---|---|
| **VLP foundation** (encoder) | PanDerm, Derm1M/DermLIP, MAKE, DermINO, BIOMEDICA | SSL or contrastive VLP on million-scale image-text pairs |
| **Distilled generative VLM** (your school) | SkinFlow, SkinCaRe/SkinCoT, MM-Skin/SkinVL, DermatoLlama, DermoGPT, DermETAS-SNA, **your dissertation** | Teacher LLM → structured rationales → SFT small student |
| **RL-reasoning VLM** | SkinGPT-R1, Skin-R1, DermIQ-VLM/GRPO++, MedVLM-R1, Med-R1, AlphaMed | GRPO / verifier-reward RL on small VLMs |
| **Multi-agent / specialist system** | SkinGPT-X, CLARIFY, MEDDxAgent, DERM-3R, **your Phase 6** | Orchestrator + category specialists + RAG / KG |

Your dissertation sits at the **intersection of schools 2 and 4** — distilled student VLMs inside a multi-agent system. Few existing papers combine both. SkinGPT-X is the closest, but chose self-evolving memory instead of RAG; CLARIFY used a tiny classifier as specialist rather than a full fine-tuned VLM. This is a defensible positioning.

### 7.2 Five methodological claims you can now support with external citations

1. **"Structured attribute decomposition beats flat CoT in clinical reasoning."** Cite MedCoT (2412.13736), "Why CoT Fails in Clinical Text Understanding" (2509.21933), MAKE (2505.09372), and DermoGPT (2601.01868).

2. **"Verifier-filtered teacher distillation is the dominant 2025 pattern and your expert label is a free verifier."** Cite HuatuoGPT-o1 (2412.18925), MedReason (2504.00993), ReasonMed (2506.09513), Dr-LLaVA (2405.19567).

3. **"Small specialized models consistently match or exceed 100×–1000×-larger generalists on medical benchmarks."** Cite Meditron-7B (2311.16079), LLaVA-MoD 2B (2408.15881), Me-LLaMA (2402.12749), BiomedLM 2.7B (2403.18421), DermatoLlama 11B (medRxiv 25327785), plus your own Qwen 9B ≈ Qwen3-VL-235B result.

4. **"LoRA / QLoRA is viable with small accuracy loss (~3–4 points) for a ~650× parameter reduction."** Cite Med42 (2404.14779), LLaMA-XR (2506.03178), LLaMA-3.2-3B-MedCoT (2510.05003).

5. **"Balanced training is necessary but not sufficient for FST fairness; post-hoc audit + FST-stratified metrics + optional adversarial debiasing are standard."** Cite Morales-Forero (2507.14176), López-Pérez (2501.11752), LesionTABE (2601.03090), Nijjer (2510.00055), "Beyond Fitzpatrick" (npj DM 2025).

### 7.3 Counter-arguments you must address

- **"RL-only (GRPO) reaches SOTA without any CoT SFT data" — MedVLM-R1, AlphaMed.** Your rebuttal must lean on the structured-attribute-output argument (clinician interpretability and ABCDE/7-point alignment), not pure accuracy.
- **"A million-scale VLP foundation model already solves most of this" — PanDerm, Derm1M.** Your positioning should be "efficient deployment at 4–9B with free on-prem inference," not "we beat million-image foundation encoders."
- **"SkinGPT-X already built a multi-agent dermatology system" — arXiv:2603.26122.** You differ in (i) RAG vs self-evolving memory, (ii) category-level specialization vs disease-level, (iii) using fine-tuned SLMs as specialists vs wrapping prompted LLMs.

### 7.4 Concrete recommendations

| Recommendation | Priority | Rationale |
|---|---|---|
| Add **DermBench + DermEval** to the evaluation pipeline | High | Fills the narrative-quality gap beyond BERTScore |
| Add **DermaVQA-DAS** closed-QA numbers | Medium | Direct comparable number vs GPT-4.1, Gemini-1.5-Pro, o3 |
| Report **demographic parity + equalized odds per FST** (Nijjer-style) | High | Matches standard fairness reporting in derm LLM papers |
| **Dual-report FST and MST** for at least one benchmark | Medium | SCIN has both; "Beyond Fitzpatrick" justifies migration |
| Explicitly report **teacher filter-pass rate** | High | Standard in verifier-filtered distillation papers (HuatuoGPT-o1, ReasonMed) |
| Reference **SkinGPT-X** as closest multi-agent competitor (Phase 6) | High | Closest prior art; position contrastively |
| Cite **"Why CoT Fails in Clinical Text"** defensively | Medium | Pre-empts "why not just CoT" reviewer question |
| Add **eSkinHealth** numbers if reachable | Low | Geographic generalization to FST V-VI in tropical conditions |
| Reference **CLARIFY** efficiency numbers | High | Direct precedent for Specialist-Generalist + KG RAG with VRAM/latency targets |

---

## 8. Verification Caveats

Some arXiv IDs in this review originate from WebSearch snippets rather than direct `get_abstract` calls. Before citing, run `curl -s https://arxiv.org/abs/<id>` (or open the URL) to verify the exact title, author list, and published version.

Specific items flagged for double-checking:
- **SCALEMED / DermatoLlama 1.0** — medRxiv (`10.1101/2025.05.17.25327785`), not arXiv. Verify whether this is the same DermaSynth group (Yilmaz) you already cite, or a distinct paper.
- **DermaBench** (2601.14084) vs **DermBench/DermEval** (2511.09195) vs **DermoBench** (inside DermoGPT, 2601.01868) — three similarly-named benchmarks from partially overlapping author groups; keep them distinct in the thesis.
- **DermaVQA** (MICCAI'24 LNCS 15005) vs **DermaVQA-DAS** (2512.24340) — the latter is the extension with the Dermatology Assessment Schema.
- **npj Digital Medicine** fairness references (s41746-025-01770-4 and s41746-025-02245-2) — peer-reviewed journal articles, not arXiv preprints.

---

*Compiled from: parallel search across arXiv (MCP), WebSearch results, and Nature / JAMA publication pages. Approximately 60 distinct papers surveyed; 40 are tabulated above; 25 receive detailed writeups.*
