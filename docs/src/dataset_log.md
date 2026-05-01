# Dataset Log - Dermatology Reasoning Think v2

Date: 2026-04-29

This note documents the construction of `derm-reasoning-think-v2`, the cleaned
visible-reasoning successor to `danielfdias98/derm-reasoning-full-reasoning`.
It is intended as thesis-facing documentation for the dataset preparation stage.

## Purpose

The original full-reasoning dataset contained structured dermatology answers,
but the free-form `reasoning` field sometimes included non-visible clinical
claims such as symptoms, duration, palpation findings, laboratory confirmation,
dermoscopy details, or patient history. Those details are clinically relevant
when available, but they are not reliably inferable from a single ordinary
photograph. Training a VLM on such text can teach the model to justify an image
with facts that were not actually seen.

The v2 dataset keeps the original ground-truth labels and structured answer
fields, but rebuilds the reasoning section into a visible-evidence format:

- visual observations first;
- differential reasoning second;
- explicit limitations last;
- no copied raw `reasoning` text from the original dataset.

The goal is not merely to format the data as chain-of-thought. The goal is to
make the reasoning faithful to the visual evidence available to the model.

## Source Data

Source directory:

```text
data/fine_tune/full_reasoning
```

Input files:

- `train.jsonl`
- `val.jsonl`

Each source row contains a user message with an image path and an assistant
message containing JSON dermatology fields. The v2 build script parses those
assistant JSON responses, validates image references, computes image hashes, and
writes a flat Hugging Face friendly export.

The original local source files were not overwritten.

## Output Dataset

Local export directory:

```text
dataset_export/derm-reasoning-think-v2
```

Target Hub repository:

```text
danielfdias98/derm-reasoning-think-v2
```

The dataset schema is:

- `image`: embedded image path for Hub conversion;
- `instruction`: original user instruction;
- `response`: cleaned `<think>...</think><answer>...</answer>` response;
- `image_id`: filename stem;
- `class`: original class folder label;
- `source`: attributed source dataset;
- `content_hash`: SHA-256 hash of the image bytes;
- `quality_flags`: deterministic cleanup flags.

The current local v2 export contains:

| Metric | Count |
|---|---:|
| Source rows | 28,486 |
| Input train rows | 25,637 |
| Input validation rows | 2,849 |
| Kept train rows | 22,528 |
| Kept validation rows | 2,539 |
| Total kept rows | 25,067 |
| Quarantined rows | 3,419 |
| Duplicate image-hash groups detected | 506 |
| Post-clean train/validation duplicate hashes | 0 |
| Source classes observed | 337 |
| Classes retained after quarantine | 319 |

The public Hub repository was created as a new v2 dataset rather than an
overwrite of the original dataset. At the time of this log, the newest local
export above is the authoritative count because refreshing the Hub copy requires
a valid write token.

## Cleaning Rules

Rows were quarantined rather than silently deleted. The local audit artifacts
are:

```text
dataset_export/derm-reasoning-think-v2/audit/summary.json
dataset_export/derm-reasoning-think-v2/audit/quarantine.jsonl
dataset_export/derm-reasoning-think-v2/audit/duplicate_groups.jsonl
```

Quarantine counts:

| Reason | Rows |
|---|---:|
| `low_confidence` | 1,827 |
| `hard_dermoscopy` | 744 |
| `duplicate_conflicting_class` | 330 |
| `hard_nonclinical_diagnostic_media` | 298 |
| `duplicate_same_split_noncanonical` | 260 |
| `duplicate_cross_split_train_leak` | 78 |

The main rules were:

- quarantine `confidence == "low"`;
- quarantine hard dermoscopy rows, using signals such as `dermoscopy`,
  `dermoscopic`, `under magnification`, `pigment network`, and
  `reticular pigment network`;
- quarantine non-clinical diagnostic media such as agar plates, cultures,
  microscopy slides, histology references, and other laboratory-only media;
- compute SHA-256 hashes for all referenced image files;
- quarantine exact duplicate image hashes with conflicting class labels;
- remove exact train/validation leakage by keeping the validation row and
  quarantining matching training rows;
- keep one canonical row per same-label duplicate group, preferring high
  confidence, then source priority, then lowest original row index.

The source priority for duplicate tie-breaking was:

```text
pad_ufes > scin > skincap > dermnet_nz > kaggle_dermnet
```

## Response Format

Each kept row is rewritten into:

```text
<think>
Visible evidence:
- Morphology: ...
- Color: ...
- Texture: ...
- Border: ...
- Distribution: ...

Differential reasoning:
- ...

Limitations:
- Symptoms, duration, tenderness, palpation findings, lab confirmation,
  dermoscopy, and patient history are not inferable from the image alone unless
  explicitly visible or provided.
</think>
<answer>
{"diagnosis": "...", "category": "...", "confidence": "...", ...}
</answer>
```

The `<answer>` block remains machine-readable JSON. The field order is
normalized to:

```text
diagnosis, category, confidence, observation, morphology, color, texture,
border, distribution, differentials
```

The diagnosis is anchored to the original class label. If the original assistant
JSON diagnosis disagrees with the row class, the v2 builder re-anchors the
diagnosis to the class label and adds a quality flag.

## Reasoning Improvements

The v2 reasoning is deliberately narrower than the original free-form reasoning.
It does not attempt to infer hidden clinical context. Instead, it teaches a
model to follow this sequence:

1. describe visible morphology, color, texture, border, and distribution;
2. compare the visible pattern with plausible alternatives;
3. separate what can be seen from what would require history, examination,
   dermoscopy, or laboratory confirmation;
4. produce a structured JSON answer.

This is expected to improve model behavior in three ways.

First, it reduces hallucinated medical explanations. A model trained on hidden
claims can learn to say that a rash is painful, itchy, acute, chronic, biopsy
confirmed, or dermoscopically patterned even when the image alone cannot support
those statements. The v2 format makes the uncertainty explicit.

Second, it improves supervision density. Every row teaches several visual
subtasks, not just a class label: morphology, color, texture, border,
distribution, differential diagnosis, and confidence.

Third, it aligns the response format with evaluation. Because the final answer
is valid JSON inside `<answer>`, downstream scoring can measure diagnosis,
category, confidence, and differential structure separately from the visible
reasoning.

## Validation

The v2 builder includes focused tests for:

- exactly one `<think>` and one `<answer>` block;
- valid JSON inside `<answer>`;
- hidden-fact phrase removal from final response text;
- low-confidence quarantine;
- train/validation duplicate image-hash leakage prevention;
- duplicate conflicting-label quarantine;
- deterministic output on fixed inputs.

The local strict scan of the final export found:

| Check | Result |
|---|---:|
| Train rows scanned | 22,528 |
| Validation rows scanned | 2,539 |
| Empty differential rationales | 0 |
| Cross-split image-hash overlap | 0 |
| Response parse errors | 0 |

The broader repository test suite was not fully green for unrelated pre-existing
fine-tuning import issues, but the dataset builder tests passed.

## Research Grounding

The v2 dataset design responds to four converging arguments in the recent
literature on chain-of-thought (CoT) supervision for medical vision-language
models: that direct dermatology-CoT precedents already favour structured
templates over free-form prose; that free-form CoT is empirically unfaithful
in ways that are particularly dangerous in clinical settings; that explicit
structural cues function as supervision peaks under fine-tuning; and that
data-quality auditing dominates raw size for fine-tuning data.

### Direct precedents in dermatology and medical chain-of-thought

The `<think>...</think>` envelope used in v2 follows the convention introduced
by [MedVLM-R1](https://arxiv.org/abs/2502.19634) (Pan et al., 2025), with an
important methodological caveat: that paper argues supervised fine-tuning
"often suffers from overfitting to training distributions and fails to foster
genuine reasoning" and therefore uses reinforcement learning rather than SFT.
v2 responds to this concern not by abandoning SFT, but by constraining the
supervision signal itself: rather than train on free-form narrative reasoning
vulnerable to memorisation, v2 supervises a structured visible-evidence
template anchored to fields the model can verify against the image.

[SkinCaRe / SkinCoT](https://arxiv.org/abs/2405.18004) (Shen et al., 2024) is
the closest published precedent for dermatology CoT: 3,041 dermatology images
paired with clinician-verified hierarchical CoT, "rigorously evaluated against
six quality criteria and iteratively refined". The v2 audit rules play an
analogous quality-gating role at scale, although deterministic rather than
clinician-verified. [Skin-R1](https://arxiv.org/abs/2511.14900) and
[VL-MedGuide](https://arxiv.org/abs/2508.06624) report that *structural
compliance* with a fixed reasoning template is a load-bearing evaluation
metric, distinct from diagnostic accuracy; the v2 schema (Visible evidence →
Differential reasoning → Limitations → JSON answer) is designed to be scorable
along the same dimension. [DermoGPT](https://arxiv.org/abs/2601.01868) imposes
the discipline that "models must first commit to morphology before predicting
any disease label", which the v2 ordering enforces by placing morphology,
colour, texture, border, and distribution before the differential and answer.
[SkinGPT-R1 / DermCoT](https://arxiv.org/abs/2511.15242) treats CoT
supervision quality, rather than scale, as the bottleneck and constructs
DermCoT as a separately curated corpus.
[MedCLM](https://arxiv.org/abs/2510.04477),
[ClinCoT](https://arxiv.org/abs/2603.01124), and
[MedEyes](https://arxiv.org/abs/2511.22018) further argue that medical CoT
must be linked to localised visual evidence rather than text-centric
narrative, which motivates the v2 "Visible evidence" subsection.
[DermaBench](https://arxiv.org/abs/2601.14084) frames clinician-annotated
structured reasoning as the standard for evaluation, implying that downstream
training data should match that distribution.

### Chain-of-thought faithfulness and hallucination

[Faithful CoT](https://arxiv.org/abs/2301.13379) (Lyu et al., 2023) and
[Measuring Faithfulness in CoT](https://arxiv.org/abs/2307.13702) (Lanham et
al., 2023) establish that free-form CoT is not a reliable explanation of how
a model arrives at its answer, and that this risk grows rather than shrinks
with model capability.
[Chain-of-Thought Reasoning In The Wild Is Not Always Faithful](https://arxiv.org/abs/2503.08679)
(Arcuschin et al., 2025) shows that production models exhibit "implicit
post-hoc rationalisation" — fluent CoT that justifies a decision the model
has already made. [Streaming Hallucination Detection in Long CoT](https://arxiv.org/abs/2601.02170)
demonstrates that hallucinations in long CoT "often emerge subtly and
propagate across reasoning steps", making prose-heavy reasoning the
high-risk surface area.
[Consistent but Dangerous](https://arxiv.org/abs/2603.20985) (Sadanandan &
Behzadan, 2026) targets medical VLMs directly: a model can reach 99.6%
accuracy and yet ignore the image, and LoRA fine-tuning sometimes shifts
samples into this "Dangerous" quadrant rather than out of it. The v2
"Limitations" clause is an explicit response to this literature: the model is
taught to enumerate what would require history, examination, dermoscopy, or
laboratory confirmation rather than confabulate it.
[Hybrid-Code v2](https://arxiv.org/abs/2512.23743) and the survey on
[Mitigating Hallucinations in LLMs for Healthcare](https://www.embs.org/jbhi/wp-content/uploads/sites/18/2025/11/Mitigating-Hallucinations-in-Large-Language-Models-for-Healthcare-Towards-Trustworthy-Medical-AI.pdf)
(IEEE JBHI, 2025) catalogue structured prompts and chain-of-thought
verification as recommended hallucination mitigations for safety-critical
clinical AI.

### Structure and supervision

[Hopfieldian View of CoT](https://arxiv.org/abs/2410.03595) (Hu et al., 2024)
and [π² Structure-Originated Reasoning Data](https://arxiv.org/abs/2604.05114)
argue that low-dimensional, structured reasoning representations are more
robust under SFT than free-form prose.
[Thinking Tokens are Information Peaks](https://arxiv.org/abs/2506.02867)
(Qian et al., 2025) shows that explicit reasoning markers correspond to
mutual-information peaks where prediction error drops; the v2 section headers
("Visible evidence:", "Differential reasoning:", "Limitations:") are
intended to play this role.
[Description-then-Decision](https://arxiv.org/abs/2311.09193) (Wu et al.,
2023) reports a 50% boost on probing tasks from describe-first,
decide-second prompting, mirrored in the v2 ordering.

### Data quality

[Towards Reliable Dermatology Evaluation Benchmarks](https://arxiv.org/abs/2309.06961)
(Gröger et al., 2023) audits six widely-used dermatology datasets and finds
sufficient label noise and near-duplicates to invalidate benchmark trust; the
v2 quarantine pipeline is a deterministic analogue of that protocol applied
to training data. [CPRet](https://arxiv.org/abs/2505.12925) shows that
train/test duplication inflates measured accuracy, which the v2 export
defends against by verifying post-clean cross-split image-hash overlap of
zero. [Fine-T2I](https://arxiv.org/abs/2602.09439) reports that aggressive
filtering (95% of candidate samples removed) is the dominant lever in
fine-tuning data preparation, supporting the v2 quarantine of 3,419 rows
(approximately 12% of the source) over the alternative of admitting all rows.

### Foundational and dermatology-specific lineage

These references are retained from prior versions of this log:

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
  (Wei et al., 2022) — original motivation for intermediate reasoning traces.
- [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923)
  — extension of CoT to vision-language settings.
- [LLaVA-Med](https://arxiv.org/abs/2306.00890) — biomedical
  instruction-tuning lineage.
- [SkinGPT-4](https://www.nature.com/articles/s41467-024-50043-3) —
  dermatology multimodal training at broad disease scope.
- [SkinFlow](https://arxiv.org/abs/2601.09136) — structured dermatology
  captioning as a training signal for open dermatological diagnosis.
- [Disparities in Dermatology AI Performance](https://pmc.ncbi.nlm.nih.gov/articles/PMC9374341/)
  — skin-tone fairness motivation.
- [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355)
  — visual-grounding checks for VLMs.

The methodological position is that no reasoning trace should be treated as
automatically trustworthy. v2 inherits the chain-of-thought *form* but
rebuilds the *content* as visible-evidence reasoning supervised by
deterministic rules, audited duplicates, and an explicit limitations clause.

## Comparison with the `derm-reasoning-cot` Variant

The v2 dataset shares its image set, ground-truth labels, and source
attribution with [`danielfdias98/derm-reasoning-cot`](https://huggingface.co/datasets/danielfdias98/derm-reasoning-cot).
The two variants differ only in (i) assistant-turn formatting and (ii) which
rows survive cleaning. This makes them a controlled pair for a format
ablation rather than independent corpora.

| Property | `derm-reasoning-cot` | `derm-reasoning-think-v2` |
|---|---|---|
| Total rows | 28,486 | 25,067 (local) / 25,331 (Hub) |
| `<think>` content | Free-form narrative prose generated from the original `reasoning` field | Deterministically rebuilt from structured fields; original `reasoning` not copied |
| Structural sections | None | Visible evidence / Differential reasoning / Limitations |
| Anti-hallucination clause | None | Explicit limitations clause in every example |
| Quality audit | None reported | 3,419 rows quarantined with reason codes |
| Cross-split image-hash leakage | Not reported | 0 (verified post-clean) |
| Audit artifacts | None | `audit/{summary.json, quarantine.jsonl, duplicate_groups.jsonl}` |
| Intended role | Format ablation against the structured variant | Primary SFT corpus |

For supervised fine-tuning targeting trustworthy clinical reasoning, v2
supplies the four properties the medical-CoT literature consistently
identifies as load-bearing: visible-evidence grounding, an explicit
limitations clause, deterministic structure, and an audit trail. The `cot`
variant is retained as a controlled comparator: holding all hyperparameters
fixed except the assistant-turn format isolates the contribution of
structured reasoning supervision from the contribution of cleaning, since
both variants share the same underlying images and labels. This design
follows the format-ablation logic of
[Description-then-Decision](https://arxiv.org/abs/2311.09193) and
[VL-MedGuide](https://arxiv.org/abs/2508.06624).

The 3,419 rows present in `cot` but absent from v2 are not lost capacity;
they are the rows the audit identified as low-confidence (1,827), hard
dermoscopy outside the photographic-image scope (744), conflicting-class
duplicates (330), non-clinical media (298), same-split noncanonical
duplicates (260), or cross-split leakage (78). The dermatology benchmark
audit by [Gröger et al.](https://arxiv.org/abs/2309.06961) and the
fine-tuning data preparation report
[Fine-T2I](https://arxiv.org/abs/2602.09439) both indicate that aggressive
removal of low-quality samples is the dominant lever in fine-tuning data
preparation, and that these +3,419 rows are appropriately excluded from the
primary SFT corpus.

## Model Compatibility

v2 is designed to be model-agnostic. It is compatible with the four base
models in the dissertation campaign — Qwen 3.5 4B (VL), Qwen 3.5 9B (VL),
Gemma 4 E2B, and Gemma 4 E4B — although with distinct integration paths.

### Qwen 3.5 4B and 9B (vision-language)

Both Qwen 3.5 4B and 9B are unified vision-language models with native
support for chain-of-thought reasoning via `<think>...</think>` tokens.
According to the
[Qwen 3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune),
thinking is enabled or disabled at chat-template construction time via
`tokenizer.apply_chat_template(..., enable_thinking=True/False)`. Two
integration approaches are valid:

1. *Native thinking mode* (`enable_thinking=True`): the v2 `<think>` and
   `</think>` strings align with Qwen's native reasoning channel.
2. *Plain-text mode* (`enable_thinking=False`): the literal `<think>...</think>`
   tags are treated as ordinary string content within the assistant turn.

The dissertation campaign uses approach 2 for cross-model fairness: the
literal `<think>` and `<answer>` tags are part of the assistant content, not
delimited by a model-specific thinking channel. This guarantees an identical
supervision signal across architectures, which is required for the
factorial-design experiments on the four base models.

### Gemma 4 E2B and E4B (vision-language)

Both E2B and E4B are multimodal-capable. The
[Gemma 4 fine-tuning guide](https://unsloth.ai/docs/models/gemma-4/train)
lists two distinct chat templates: `gemma-4` (non-thinking) and
`gemma-4-thinking` (thinking-enabled, with native channel tokens). The
[Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
notes that for E2B and E4B specifically, disabling thinking does not produce
empty thought blocks, unlike larger Gemma 4 variants — a useful property
when the v2 plain-text integration is used.

The v2 dataset is compatible with the `gemma-4` (non-thinking) template
under the same plain-text integration as Qwen 3.5: the literal
`<think>...</think>` tags appear as plain assistant content. The
`gemma-4-thinking` template is *not* recommended for v2, because Gemma's
native thinking channel uses different special tokens than the v2 literal
strings; combining them would interleave Gemma's native reasoning channel
with the v2 string content and produce ambiguous supervision.

### Loss masking and supervision target

Independent of model family, supervision is loss-masked (`labels = -100`) on
the user turn and computed only on the assistant turn beginning at `<think>`.
Both Qwen 3.5 and Gemma 4 chat templates expose the assistant-content
boundary required to apply this mask. A multimodal collator that walks the
chat template and emits the user-turn-prefix length is sufficient.

### Summary of compatibility

| Model | Native thinking channel | Recommended chat-template setting | v2 compatibility |
|---|---|---|---|
| Qwen 3.5 4B (VL) | Yes (native `<think>` tokens) | `enable_thinking=False`, train on literal tags | Compatible |
| Qwen 3.5 9B (VL) | Yes (native `<think>` tokens) | `enable_thinking=False`, train on literal tags | Compatible |
| Gemma 4 E2B | Yes (Gemma native channel) | `gemma-4` (non-thinking) | Compatible |
| Gemma 4 E4B | Yes (Gemma native channel) | `gemma-4` (non-thinking) | Compatible |

## Limitations

The v2 build is deterministic and reproducible, but it is not a full
dermatologist review. It can remove known hidden-claim patterns and duplicate
leakage, but it cannot independently verify every image-text pair for visual
accuracy. For example, a row can still contain an inaccurate visible descriptor
if the original structured fields were wrong but did not trigger a deterministic
sanitization rule.

Therefore, v2 should be understood as a cleaned baseline for visible-thinking
fine-tuning. A v3 dataset should add image-aware review: each row should be
checked against the actual photograph, and the visible evidence should be
rewritten when the text overstates, omits, or contradicts what is visible.

## Reproducibility

The v2 dataset is built with:

```bash
python scripts/build_think_v2_dataset.py \
  --source-dir data/fine_tune/full_reasoning \
  --out-dir dataset_export/derm-reasoning-think-v2
```

A sample dry run can be produced with:

```bash
python scripts/build_think_v2_dataset.py \
  --source-dir data/fine_tune/full_reasoning \
  --out-dir dataset_export/derm-reasoning-think-v2-sample \
  --limit 20 \
  --dry-run
```
