# Chain-of-Thought Format Decision for Dermatology SFT and GRPO

Date: 2026-05-01

This note explains the final Chain-of-Thought (CoT) format I recommend for the
next dermatology reasoning dataset, based on `OpusResearch.md`, the current
`derm-reasoning-think-v2` dataset log, and additional research into 2026 CoT,
medical VLM, and GRPO papers.

The short version:

Use a medium-length, structured, visible-evidence-first CoT with a machine-readable
JSON answer. Do not use free-form CoT, very long R1-style CoT, or diagnosis-first
justifications. The format should be designed so that GRPO can reward not only
final diagnosis correctness, but also format validity, visual grounding,
reasoning-to-answer consistency, differential quality, and anti-hallucination
behavior.

## Final Recommended Format

```text
<think>
Visible evidence:
- Morphology: ...
- Color: ...
- Surface/texture: ...
- Border: ...
- Distribution/location: ...
- Arrangement: ...

Differential reasoning:
- <candidate_1>: supported by ...; less supported because ...
- <candidate_2>: supported by ...; less supported because ...
- <candidate_3>: supported by ...; less supported because ...

Most likely diagnosis:
- The visible pattern is most consistent with <diagnosis>.

Uncertainty / limitations:
- Image-only assessment; symptoms, duration, tenderness, palpation, dermoscopy,
  pathology, laboratory confirmation, treatment response, and clinical history
  are not available unless explicitly provided.
</think>
<answer>
{
  "diagnosis": "<canonical_class>",
  "confidence": "low|medium|high",
  "differential": [
    {
      "diagnosis": "<canonical_class_1>",
      "supported_by": ["<visible evidence>", "<visible evidence>"],
      "less_supported_by": ["<visible limitation or contradiction>"]
    },
    {
      "diagnosis": "<canonical_class_2>",
      "supported_by": ["<visible evidence>"],
      "less_supported_by": ["<visible limitation or contradiction>"]
    },
    {
      "diagnosis": "<canonical_class_3>",
      "supported_by": ["<visible evidence>"],
      "less_supported_by": ["<visible limitation or contradiction>"]
    }
  ],
  "visual_evidence": {
    "morphology": "<short visible description>",
    "color": "<short visible description>",
    "surface_texture": "<short visible description>",
    "border": "<short visible description>",
    "distribution_location": "<short visible description>",
    "arrangement": "<short visible description>"
  },
  "limitations": [
    "image_only",
    "no_clinical_history",
    "no_dermoscopy",
    "no_pathology",
    "no_lab_confirmation"
  ]
}
</answer>
```

Recommended constraints:

- Keep the final diagnosis as the first candidate in `differential`.
- Use canonical dataset class labels in `diagnosis` and `differential[*].diagnosis`.
- Use three differential candidates when plausible; use two if a third would be
  artificial.
- Every `supported_by` and `less_supported_by` item must be visually grounded.
- If a feature is not visible, write `not visible` or `not assessable from image
  alone` instead of inventing clinical context.
- Keep the reasoning medium length. The target is concise clinical reasoning, not
  exhaustive long-CoT exploration.
- The `<answer>` block must be valid JSON and must not contain markdown comments.

## Why This Format

### 1. Dermatology is a visual-grounding problem before it is a reasoning problem

The v2 dataset already moved in the right direction: it rewrites the answer into
visible evidence, differential reasoning, limitations, and JSON. That matters
because the original full-reasoning data could contain hidden clinical claims
such as symptoms, duration, history, palpation findings, lab confirmation, or
dermoscopy. Those facts can be clinically useful when provided, but they are not
usually inferable from one ordinary clinical photograph.

The final v3 format keeps the v2 idea but makes it more rewardable:

- visible evidence is split into fixed dermatology fields;
- the differential is structured as objects instead of plain strings;
- the JSON answer repeats the visual evidence in a parseable form;
- limitations are encoded as stable tokens that can be checked automatically.

This gives SFT a clean target and gives GRPO a set of measurable behaviors.

### 2. Description-before-diagnosis matches the medical VLM literature

Several dermatology and medical VLM papers support a describe-then-decide
structure:

- [SkinCaRe / SkinCoT](https://arxiv.org/abs/2405.18004) introduced clinician
  verified dermatology CoT cases and emphasizes medically described visual
  reasoning.
- [VL-MedGuide](https://arxiv.org/abs/2508.06624) uses dermatologic visual
  concept perception before disease reasoning.
- [DermoGPT](https://arxiv.org/abs/2601.01868) is directly aligned with this
  recommendation: morphology-anchored dermatology instruction data, followed by
  SFT and visual-inference-consistency RL.
- [DermaBench](https://arxiv.org/abs/2601.14084) shows what dermatologist
  annotation actually tracks: diagnosis, anatomic site, morphology,
  distribution, surface features, color, image quality, and narrative summary.

That is why the final format begins with morphology, color, surface/texture,
border, distribution/location, and arrangement.

### 3. GRPO should not reward answer accuracy alone

The strongest GRPO lesson is that answer-only rewards can improve final accuracy
while damaging the reasoning trace.

- [GRPO-CARE](https://arxiv.org/abs/2506.16141) finds that standard GRPO can
  improve answer accuracy while reducing logical coherence between reasoning and
  answers. It adds a consistency-aware reward to improve both.
- [Faithful GRPO](https://arxiv.org/abs/2604.08476) shows the same pattern in
  multimodal spatial reasoning: standard GRPO improves accuracy but CoT traces
  can become inconsistent or poorly grounded. Their constrained method reduces
  inconsistency from 24.5% to 1.7% and improves grounding.
- [MedEyes](https://arxiv.org/abs/2511.22018) warns that on-policy RLVR in
  medical VLMs can reinforce superficially coherent but clinically inaccurate
  reasoning paths without stronger grounding signals.

The JSON schema is therefore not cosmetic. It gives GRPO handles for:

- valid tag and JSON format;
- final diagnosis match;
- differential rank and membership;
- evidence overlap with allowed visual fields;
- consistency between `<think>` and `<answer>`;
- penalties for hidden clinical claims;
- length control.

### 4. SFT should teach the format; GRPO should sharpen it

The best training order is:

1. SFT on the curated v3 visible-reasoning dataset.
2. GRPO after SFT, using rewards that check correctness, format, grounding, and
   consistency.

This is supported by:

- [When Does RL Help Medical VLMs?](https://arxiv.org/abs/2603.01301), which
  argues that SFT expands the model's support and RL is most useful when the
  model already has non-trivial probability mass on the right answer.
- [MedVLM-R1](https://arxiv.org/abs/2502.19634), which demonstrates that RL can
  substantially improve explicit medical visual reasoning, though its authors
  are skeptical of naive SFT.
- [Skin-R1](https://arxiv.org/abs/2511.14900), which uses SFT to establish a
  grounded reasoning foundation and then RL to transfer reasoning patterns.
- [DermoGPT](https://arxiv.org/abs/2601.01868), which uses SFT followed by a
  morphology-anchored visual-inference-consistency RL objective.

For this project, SFT is still necessary because Qwen 3.5 9B and Gemma 4 E4B
need to learn the dermatology vocabulary, canonical class space, and exact output
schema before GRPO can reliably optimize behavior.

### 5. Medium CoT is safer than long CoT for this dataset

Long CoT is useful in some symbolic and mathematical tasks, but it is not the
right default for image-only dermatology diagnosis.

- [Balanced Thinking](https://arxiv.org/abs/2603.18656) argues that long
  `<think>` traces can dominate SFT loss and weaken supervision on the shorter,
  task-critical `<answer>` segment.
- [Streaming Hallucination Detection in Long CoT](https://arxiv.org/abs/2601.02170)
  frames long-CoT hallucination as an evolving state where early mistakes can
  propagate through later reasoning.
- [DiffCoT](https://arxiv.org/abs/2601.03559) proposes a more complex correction
  mechanism for error accumulation, which is interesting research but too heavy
  for the immediate v3 dataset design.

So the target should be "draft clinical reasoning": enough reasoning to force
visual observation and differential comparison, but not enough length to invite
overthinking, unsupported clinical speculation, or answer-token underweighting.

## Papers Researched and How They Affected the Decision

### Foundational CoT and faithfulness

| Paper | Main finding used here | Effect on v3 |
|---|---|---|
| [Wei et al., 2022 - Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) | Intermediate reasoning can improve complex task performance. | CoT is worth training, but the format must be task-specific. |
| [Lanham et al., 2023 - Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702) | CoT can be unfaithful; models do not always use the stated rationale to answer. | Treat rationale as a behavior to evaluate, not proof of true reasoning. |
| [Lobo et al., 2024/2025 - On the Impact of Fine-Tuning on CoT Reasoning](https://arxiv.org/abs/2411.15382) | Fine-tuning can improve task behavior while decreasing CoT faithfulness. | Add anti-hallucination rules and consistency checks. |
| [Schaeffer et al., 2023 - Invalid Logic, Equivalent Gains](https://arxiv.org/abs/2307.10573) | Logically invalid rationales can still produce performance gains. | Do not assume fluent rationale equals valid rationale. |
| [Yeo et al., 2025 - Demystifying Long CoT](https://arxiv.org/abs/2502.03373) | Long-CoT behaviors are more associated with RL dynamics than simply scaling SFT. | Use SFT as a cold start, not as a substitute for reward shaping. |

### Medical, multimodal, and dermatology CoT

| Paper | Main finding used here | Effect on v3 |
|---|---|---|
| [SkinCaRe / SkinCoT](https://arxiv.org/abs/2405.18004) | Dermatology CoT benefits from clinician-verified, hierarchical visual reasoning. | Keep structured visual fields and differential reasoning. |
| [MedVLM-R1](https://arxiv.org/abs/2502.19634) | RL can incentivize explicit medical VLM reasoning and improve accuracy. | Plan SFT -> GRPO rather than SFT-only. |
| [VL-MedGuide](https://arxiv.org/abs/2508.06624) | Dermatology diagnosis benefits from visual concept perception before reasoning. | Put visual evidence before the diagnosis. |
| [Skin-R1](https://arxiv.org/abs/2511.14900) | SFT installs grounded reasoning; RL transfers it to broader dermatology data. | Use SFT to teach the schema, then GRPO to sharpen behavior. |
| [SkinGPT-R1](https://arxiv.org/abs/2511.15242) | Structured diagnostic reports with visual findings, differential reasoning, and final diagnosis support trustworthy dermatology reasoning. | Keep the final answer report-like and structured. |
| [MedEyes](https://arxiv.org/abs/2511.22018) | Medical RLVR can reinforce superficially coherent but inaccurate reasoning unless visually guided. | Add visual grounding rewards and hidden-fact penalties. |
| [DermoGPT](https://arxiv.org/abs/2601.01868) | Morphology-anchored dermatology data plus visual-inference-consistency RL is directly useful. | Make morphology and visual consistency central to v3. |
| [DermaBench](https://arxiv.org/abs/2601.14084) | Dermatologist VQA annotations focus on morphology, distribution, surface features, color, diagnosis, and descriptions. | Use these same fields in `visual_evidence`. |
| [Step-CoT](https://arxiv.org/abs/2603.13878) | Structured, clinician-like multi-step reasoning improves medical VQA interpretability. | Keep fixed section headers instead of free-form prose. |
| [MedVR](https://arxiv.org/abs/2604.08203) | Medical visual reasoning must be grounded in visual evidence to reduce hallucination. | Reward evidence-grounded answers, not just class correctness. |
| [SkinCLIP-VL](https://arxiv.org/abs/2603.21010) | Consistency-aware visual-language alignment improves trustworthy skin cancer diagnosis. | Keep consistency and calibration as evaluation targets. |

### 2026 GRPO and CoT training papers

| Paper | Main finding used here | Effect on v3 |
|---|---|---|
| [Faithful GRPO](https://arxiv.org/abs/2604.08476) | GRPO can harm rationale consistency/grounding unless constrained. | Use multi-component rewards for consistency and grounding. |
| [Balanced Thinking](https://arxiv.org/abs/2603.18656) | Long `<think>` segments can dominate SFT loss and hurt answer quality. | Keep CoT medium length and consider answer-weighted SFT. |
| [When Does RL Help Medical VLMs?](https://arxiv.org/abs/2603.01301) | SFT expands support; RL sharpens the output distribution when the right answer is reachable. | SFT first, GRPO second. |
| [Rethinking Token-Level Policy Optimization for Multimodal CoT](https://arxiv.org/abs/2603.22847) | Successful multimodal CoT separates perceptual grounding from exploratory inference. | Separate `Visible evidence` from `Differential reasoning`. |
| [Streaming Hallucination Detection in Long CoT](https://arxiv.org/abs/2601.02170) | Long-CoT hallucinations can accumulate over the reasoning trajectory. | Avoid verbose long-CoT traces in v3. |
| [DiffCoT](https://arxiv.org/abs/2601.03559) | Iterative correction can improve robustness in multi-step CoT. | Interesting but not the best first v3 format; too complex for current dataset generation. |
| [TRL GRPOTrainer documentation](https://huggingface.co/docs/trl/grpo_trainer) | GRPO supports custom and multiple reward functions. | The schema should expose multiple rewardable properties. |

## Where Opus Was Correct

Opus was directionally correct on the main design:

- structured CoT is preferable to free-form CoT for medical visual reasoning;
- visible evidence should come before diagnosis;
- final answers should be machine-readable;
- SFT is useful as a cold start;
- GRPO/RLVR is the natural next step if the reward is designed carefully;
- CoT faithfulness is a real risk and should be acknowledged.

The strongest part of the Opus recommendation was the template family:

```text
Visible evidence -> Differential reasoning -> Most likely diagnosis -> Limitations -> JSON answer
```

That is still the right family for v3.

## Where I Would Tighten Opus' Recommendation

I would change four things.

First, do not make the differential just a list of strings. Use objects with
`diagnosis`, `supported_by`, and `less_supported_by`. This makes the differential
scorable during GRPO.

Second, do not use very long R1-style reasoning for this dataset. Dermatology
image classification is not a math proof. Long traces increase the chance of
unsupported symptoms, time course, history, palpation, biopsy, lab, or dermoscopy
claims.

Third, do not reward GRPO with diagnosis accuracy alone. That would optimize the
easiest shortcut: the right label with a plausible-looking but ungrounded
rationale.

Fourth, keep the limitations stable and explicit. A model trained for image-only
dermatology should repeatedly learn the boundary between visible evidence and
missing clinical context.

## Suggested GRPO Reward Components

The v3 schema supports this reward design:

| Reward | What it checks |
|---|---|
| `format_reward` | Exactly one `<think>` block, one `<answer>` block, valid JSON. |
| `diagnosis_reward` | `answer.diagnosis` matches the canonical class or accepted alias. |
| `differential_reward` | `answer.differential[0].diagnosis` matches the final diagnosis; top-k contains plausible alternatives. |
| `evidence_reward` | Required visual fields are present and short. |
| `grounding_reward` | Evidence in JSON is consistent with the `Visible evidence` text. |
| `consistency_reward` | `<think>` most-likely diagnosis equals JSON `diagnosis`. |
| `anti_hallucination_reward` | Penalize symptoms, duration, pain, itch, treatment response, history, family history, biopsy/pathology, KOH/lab, dermoscopy, or "confirmed diagnosis" unless explicitly provided. |
| `length_reward` | Prefer medium outputs; penalize very short missing-rationale outputs and very long overthinking outputs. |

For thesis evaluation, report diagnosis accuracy separately from rationale
quality. A correct diagnosis with an ungrounded rationale should not be treated
as fully correct.

## Practical Final Decision

For the v3 dataset, use:

- literal `<think>` and `<answer>` tags for both Qwen 3.5 9B and Gemma 4 E4B;
- fixed section headers inside `<think>`;
- visible evidence before differential reasoning;
- a ranked differential with structured evidence objects;
- a parseable JSON answer;
- stable limitation tokens;
- medium-length rationales;
- SFT first, then GRPO with multi-component rewards.

This format is the best fit because it balances three constraints at once:

1. It teaches dermatology-relevant visual reasoning during SFT.
2. It avoids the main known failure mode of CoT fine-tuning: fluent but
   unfaithful rationales.
3. It exposes enough structure for GRPO to reward correctness, grounding,
   consistency, and restraint instead of only rewarding the final class label.

