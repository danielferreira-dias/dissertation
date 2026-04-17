# Fine-Tune Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a config-driven LoRA fine-tuning pipeline at `src/fine_tune/` that can sequentially fine-tune four VLMs (MedGemma 1.5 4B, Gemma 4 E4B, Qwen 3.5 4B, Qwen 3.5 9B) on 28,486 Gemini-3.1-generated structured-reasoning entries, running on RunPod L40S.

**Architecture:** TRL `SFTTrainer` + PEFT LoRA (rank 64, α=128), vision tower frozen, bf16 + FA2, 5 epochs, `paged_adamw_8bit` for 9B only. Data prep runs once (stratified 95/5 split). Each model is a YAML config swap. LoRA adapters pushed to a private HF Hub repo at end of each run.

**Tech Stack:** Python 3.11, `transformers>=4.48`, `peft>=0.13`, `trl>=0.13`, `accelerate>=1.2`, `bitsandbytes>=0.44`, `datasets>=3.0`, `flash-attn>=2.6`, `pytest` for local tests.

**Reference spec:** [docs/superpowers/specs/2026-04-17-fine-tune-pipeline-design.md](../specs/2026-04-17-fine-tune-pipeline-design.md)

**Execution context:** Tasks 1–6, 8, 10–12 run on laptop (pure Python, no GPU). Tasks 7, 9, 13–15 are GPU-dependent and verified on a RunPod L40S pod.

---

## Task 1: Scaffold package + dev dependencies

**Files:**
- Create: `src/fine_tune/__init__.py`
- Create: `src/fine_tune/requirements-gpu.txt`
- Create: `src/fine_tune/requirements-dev.txt`
- Create: `tests/__init__.py`
- Create: `tests/fine_tune/__init__.py`
- Create: `pytest.ini`

- [ ] **Step 1: Create empty package init files**

```bash
mkdir -p src/fine_tune/configs src/fine_tune/scripts tests/fine_tune/fixtures
touch src/fine_tune/__init__.py tests/__init__.py tests/fine_tune/__init__.py
```

- [ ] **Step 2: Write `src/fine_tune/requirements-gpu.txt`**

```
# Installed by scripts/setup_pod.sh on a fresh RunPod L40S pod.
# Pins track the spec; bump as needed at runtime and commit the change.
transformers>=4.48,<5.0
peft>=0.13,<0.20
trl>=0.13,<0.30
accelerate>=1.2,<2.0
bitsandbytes>=0.44,<1.0
datasets>=3.0,<4.0
pillow>=10.0
pyyaml>=6.0
tensorboard>=2.15
sentencepiece>=0.2
# flash-attn is installed separately (--no-build-isolation) because it needs torch headers
```

- [ ] **Step 3: Write `src/fine_tune/requirements-dev.txt`**

```
# Local-only dev deps (no GPU needed). Installed on laptop for running pytest.
pytest>=8.0
pyyaml>=6.0
pillow>=10.0
```

- [ ] **Step 4: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

- [ ] **Step 5: Install dev deps locally**

Run: `pip install -r src/fine_tune/requirements-dev.txt`
Expected: successful install, no errors.

- [ ] **Step 6: Commit**

```bash
git add src/fine_tune/ tests/ pytest.ini
git commit -m "scaffold(fine_tune): package layout + dev/gpu requirements"
```

---

## Task 2: `prepare_data.py` — stratified split + schema unification

The most fragile piece. Pure data transformation — fully TDD-able without GPU.

**Files:**
- Create: `src/fine_tune/prepare_data.py`
- Create: `tests/fine_tune/test_prepare_data.py`
- Create: `tests/fine_tune/fixtures/reasoning_sample.jsonl`

- [ ] **Step 1: Write the fixture**

Create `tests/fine_tune/fixtures/reasoning_sample.jsonl` with 10 synthetic entries covering 4 classes with a mix of `label_match=true` and `label_match=false`. Paste verbatim:

```json
{"image_path": "final/train/melanoma/mel-01.jpg", "ground_truth": "melanoma", "category": "malignant", "dataset_source": "test", "observation": {"morphology": "asymmetric pigmented lesion", "color": "dark brown", "texture": "smooth", "border": "irregular", "distribution": "back"}, "reasoning": {"morphology": "m1", "color": "c1", "texture": "t1", "border": "b1", "distribution": "d1", "clinical_reasoning": "r1", "differentials": [{"condition": "nevus", "why_not": "regular"}, {"condition": "seborrheic keratosis", "why_not": "not stuck-on"}], "confidence": "high", "label_match": true}}
{"image_path": "final/train/melanoma/mel-02.jpg", "ground_truth": "melanoma", "category": "malignant", "dataset_source": "test", "observation": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d"}, "reasoning": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d", "clinical_reasoning": "r", "differentials": [{"condition": "nevus", "why_not": "x"}], "confidence": "medium", "label_match": true}}
{"image_path": "final/train/psoriasis/ps-01.jpg", "ground_truth": "psoriasis", "category": "inflammatory", "dataset_source": "test", "observation": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d"}, "reasoning": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d", "clinical_reasoning": "r", "differentials": [{"condition": "eczema", "why_not": "x"}], "confidence": "high", "label_match": true}}
{"image_path": "final/train/psoriasis/ps-02.jpg", "ground_truth": "psoriasis", "category": "inflammatory", "dataset_source": "test", "observation": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d"}, "reasoning": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d", "clinical_reasoning": "r", "differentials": [], "confidence": "low", "label_match": false, "label_match_reason": "looks more like eczema"}}
{"image_path": "final/train/psoriasis/ps-03.jpg", "ground_truth": "psoriasis", "category": "inflammatory", "dataset_source": "test", "observation": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d"}, "reasoning": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d", "clinical_reasoning": "r", "differentials": [{"condition": "eczema", "why_not": "x"}], "confidence": "medium", "label_match": true}}
{"image_path": "final/train/eczema/ec-01.jpg", "ground_truth": "eczema", "category": "inflammatory", "dataset_source": "test", "observation": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d"}, "reasoning": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d", "clinical_reasoning": "r", "differentials": [{"condition": "psoriasis", "why_not": "x"}], "confidence": "medium", "label_match": true}}
{"image_path": "final/train/eczema/ec-02.jpg", "ground_truth": "eczema", "category": "inflammatory", "dataset_source": "test", "observation": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d"}, "reasoning": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d", "clinical_reasoning": "r", "differentials": [{"condition": "psoriasis", "why_not": "x"}], "confidence": "low", "label_match": true}}
{"image_path": "final/train/eczema/ec-03.jpg", "ground_truth": "eczema", "category": "inflammatory", "dataset_source": "test", "observation": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d"}, "reasoning": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d", "clinical_reasoning": "r", "differentials": [{"condition": "psoriasis", "why_not": "x"}], "confidence": "medium", "label_match": true}}
{"image_path": "final/train/basal_cell_carcinoma/bcc-01.jpg", "ground_truth": "basal_cell_carcinoma", "category": "malignant", "dataset_source": "test", "observation": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d"}, "reasoning": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d", "clinical_reasoning": "r", "differentials": [{"condition": "scc", "why_not": "x"}], "confidence": "high", "label_match": true}}
{"image_path": "final/train/basal_cell_carcinoma/bcc-02.jpg", "ground_truth": "basal_cell_carcinoma", "category": "malignant", "dataset_source": "test", "observation": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d"}, "reasoning": {"morphology": "m", "color": "c", "texture": "t", "border": "b", "distribution": "d", "clinical_reasoning": "r", "differentials": [{"condition": "melanoma", "why_not": "x"}], "confidence": "medium", "label_match": false, "label_match_reason": "unclear"}}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/fine_tune/test_prepare_data.py`:

```python
"""Tests for src/fine_tune/prepare_data.py."""
import json
from pathlib import Path

import pytest

from fine_tune.prepare_data import (
    build_assistant_payload,
    prepare_splits,
    entry_to_chat,
)

FIXTURE = Path(__file__).parent / "fixtures" / "reasoning_sample.jsonl"


def load_fixture() -> list[dict]:
    return [json.loads(l) for l in FIXTURE.read_text().splitlines() if l.strip()]


class TestBuildAssistantPayload:
    def test_emits_unified_schema_keys(self):
        entry = load_fixture()[0]  # melanoma, label_match=true, 2 differentials
        payload = build_assistant_payload(entry)
        assert set(payload.keys()) >= {
            "diagnosis", "top_n", "confidence", "category",
            "observation", "reasoning", "differentials",
        }

    def test_diagnosis_is_prettified_ground_truth(self):
        entry = load_fixture()[0]
        payload = build_assistant_payload(entry)
        assert payload["diagnosis"] == "Melanoma"

    def test_top_n_starts_with_diagnosis(self):
        entry = load_fixture()[0]  # 2 differentials
        payload = build_assistant_payload(entry)
        assert payload["top_n"][0] == "Melanoma"
        assert len(payload["top_n"]) == 3  # diagnosis + 2 differentials

    def test_top_n_handles_zero_differentials(self):
        entry = load_fixture()[3]  # psoriasis with differentials=[]
        payload = build_assistant_payload(entry)
        assert payload["top_n"] == ["Psoriasis"]

    def test_observation_is_nested_dict(self):
        entry = load_fixture()[0]
        payload = build_assistant_payload(entry)
        assert isinstance(payload["observation"], dict)
        assert "morphology" in payload["observation"]


class TestEntryToChat:
    def test_produces_messages_structure(self):
        entry = load_fixture()[0]
        chat = entry_to_chat(entry, image_root=Path("/workspace/data/images"))
        assert "messages" in chat
        assert len(chat["messages"]) == 2
        assert chat["messages"][0]["role"] == "user"
        assert chat["messages"][1]["role"] == "assistant"

    def test_user_turn_has_image_and_text(self):
        entry = load_fixture()[0]
        chat = entry_to_chat(entry, image_root=Path("/workspace/data/images"))
        contents = chat["messages"][0]["content"]
        types = {c["type"] for c in contents}
        assert types == {"image", "text"}

    def test_image_path_is_absolute(self):
        entry = load_fixture()[0]
        chat = entry_to_chat(entry, image_root=Path("/workspace/data/images"))
        img_content = next(c for c in chat["messages"][0]["content"] if c["type"] == "image")
        assert img_content["image"].startswith("/workspace/data/images/")
        assert img_content["image"].endswith("final/train/melanoma/mel-01.jpg")

    def test_assistant_content_is_json_string(self):
        entry = load_fixture()[0]
        chat = entry_to_chat(entry, image_root=Path("/workspace/data/images"))
        payload = json.loads(chat["messages"][1]["content"])
        assert payload["diagnosis"] == "Melanoma"


class TestPrepareSplits:
    def test_separates_flagged_entries(self, tmp_path):
        train_file = tmp_path / "train.jsonl"
        val_file = tmp_path / "val.jsonl"
        audit_file = tmp_path / "audit.jsonl"
        stats = prepare_splits(
            source=FIXTURE,
            train_out=train_file,
            val_out=val_file,
            audit_out=audit_file,
            val_fraction=0.2,
            image_root=Path("/workspace/data/images"),
            seed=42,
        )
        # 10 total, 2 flagged -> 8 training entries, 2 in audit
        audit_lines = audit_file.read_text().splitlines()
        assert len(audit_lines) == 2
        train_lines = train_file.read_text().splitlines()
        val_lines = val_file.read_text().splitlines()
        assert len(train_lines) + len(val_lines) == 8
        assert stats["audited"] == 2
        assert stats["train"] + stats["val"] == 8

    def test_stratified_split_keeps_each_class_in_train(self, tmp_path):
        """Each class with >=2 non-flagged samples should appear in train."""
        train_file = tmp_path / "train.jsonl"
        val_file = tmp_path / "val.jsonl"
        audit_file = tmp_path / "audit.jsonl"
        prepare_splits(
            source=FIXTURE,
            train_out=train_file,
            val_out=val_file,
            audit_out=audit_file,
            val_fraction=0.2,
            image_root=Path("/workspace/data/images"),
            seed=42,
        )
        train_classes = set()
        for line in train_file.read_text().splitlines():
            payload = json.loads(json.loads(line)["messages"][1]["content"])
            train_classes.add(payload["diagnosis"])
        # After filtering: melanoma(2), psoriasis(2), eczema(3), bcc(1 non-flagged)
        # With 20% val fraction and stratified split, train must include all classes with >=2 samples
        assert "Melanoma" in train_classes
        assert "Psoriasis" in train_classes
        assert "Eczema" in train_classes

    def test_deterministic_with_fixed_seed(self, tmp_path):
        """Two runs with the same seed produce byte-identical outputs."""
        out_a = tmp_path / "a"; out_a.mkdir()
        out_b = tmp_path / "b"; out_b.mkdir()
        for out_dir in (out_a, out_b):
            prepare_splits(
                source=FIXTURE,
                train_out=out_dir / "train.jsonl",
                val_out=out_dir / "val.jsonl",
                audit_out=out_dir / "audit.jsonl",
                val_fraction=0.2,
                image_root=Path("/workspace/data/images"),
                seed=42,
            )
        assert (out_a / "train.jsonl").read_bytes() == (out_b / "train.jsonl").read_bytes()
        assert (out_a / "val.jsonl").read_bytes() == (out_b / "val.jsonl").read_bytes()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/fine_tune/test_prepare_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fine_tune.prepare_data'`

- [ ] **Step 4: Implement `src/fine_tune/prepare_data.py`**

```python
"""Convert reasoning.jsonl into stratified train/val splits in chat format.

Run once per dataset. Output consumed by `src/fine_tune/train.py`.

Usage:
    python -m fine_tune.prepare_data \
        --source data/reasoning/reasoning.jsonl \
        --train-out data/reasoning/train.jsonl \
        --val-out data/reasoning/val.jsonl \
        --audit-out data/reasoning/audit.jsonl \
        --image-root /workspace/data/images \
        --val-fraction 0.05 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


USER_PROMPT = "Diagnose this skin condition with structured reasoning."


def _prettify(label: str) -> str:
    return label.replace("_", " ").title()


def build_assistant_payload(entry: dict[str, Any]) -> dict[str, Any]:
    """Emit the unified assistant-response schema from a reasoning entry.

    Schema fields: diagnosis, top_n, confidence, category, observation,
    reasoning, differentials. See spec §3.2.
    """
    reasoning = entry.get("reasoning", {}) or {}
    observation = entry.get("observation", {}) or {}
    diagnosis = _prettify(entry["ground_truth"])
    differentials = reasoning.get("differentials") or []

    top_n = [diagnosis] + [d.get("condition", "").strip().title()
                            for d in differentials
                            if d.get("condition", "").strip()]

    return {
        "diagnosis": diagnosis,
        "top_n": top_n,
        "confidence": reasoning.get("confidence", "medium"),
        "category": entry.get("category", ""),
        "observation": {
            "morphology": observation.get("morphology", ""),
            "color": observation.get("color", ""),
            "texture": observation.get("texture", ""),
            "border": observation.get("border", ""),
            "distribution": observation.get("distribution", ""),
        },
        "reasoning": reasoning.get("clinical_reasoning", ""),
        "differentials": differentials,
    }


def entry_to_chat(entry: dict[str, Any], image_root: Path) -> dict[str, Any]:
    """Build a single chat-format training example."""
    abs_image = str((image_root / entry["image_path"]).as_posix())
    payload = build_assistant_payload(entry)
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": abs_image},
                {"type": "text", "text": USER_PROMPT},
            ]},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ]
    }


def prepare_splits(
    source: Path,
    train_out: Path,
    val_out: Path,
    audit_out: Path,
    val_fraction: float,
    image_root: Path,
    seed: int,
) -> dict[str, int]:
    """Load reasoning.jsonl, filter flagged, stratified-split to train/val.

    Returns a summary dict with counts.
    """
    entries = [json.loads(l) for l in source.read_text().splitlines() if l.strip()]

    audited: list[dict] = []
    kept: list[dict] = []
    for e in entries:
        if (e.get("reasoning") or {}).get("label_match") is False:
            audited.append(e)
        else:
            kept.append(e)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for e in kept:
        by_class[e["ground_truth"]].append(e)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for label in sorted(by_class.keys()):
        bucket = by_class[label][:]
        rng.shuffle(bucket)
        n_val = max(1, int(round(len(bucket) * val_fraction))) if len(bucket) >= 2 else 0
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])

    # Final shuffle so the trainer doesn't see class-grouped batches
    rng.shuffle(train)
    rng.shuffle(val)

    for out_path, rows in ((train_out, train), (val_out, val)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for e in rows:
                f.write(json.dumps(entry_to_chat(e, image_root)) + "\n")

    audit_out.parent.mkdir(parents=True, exist_ok=True)
    with audit_out.open("w") as f:
        for e in audited:
            f.write(json.dumps(e) + "\n")

    return {
        "total": len(entries),
        "audited": len(audited),
        "train": len(train),
        "val": len(val),
        "classes": len(by_class),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/reasoning/reasoning.jsonl"))
    p.add_argument("--train-out", type=Path, default=Path("data/reasoning/train.jsonl"))
    p.add_argument("--val-out", type=Path, default=Path("data/reasoning/val.jsonl"))
    p.add_argument("--audit-out", type=Path, default=Path("data/reasoning/audit.jsonl"))
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--image-root", type=Path, default=Path("/workspace/data/images"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    stats = prepare_splits(
        source=args.source,
        train_out=args.train_out,
        val_out=args.val_out,
        audit_out=args.audit_out,
        val_fraction=args.val_fraction,
        image_root=args.image_root,
        seed=args.seed,
    )
    print(f"Source entries:     {stats['total']}")
    print(f"Audited (flagged):  {stats['audited']}")
    print(f"Train examples:     {stats['train']}")
    print(f"Val examples:       {stats['val']}")
    print(f"Classes:            {stats['classes']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/fine_tune/test_prepare_data.py -v`
Expected: all 10 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fine_tune/prepare_data.py tests/fine_tune/
git commit -m "feat(fine_tune): prepare_data.py with stratified split + schema unification"
```

---

## Task 3: Annotated YAML template (`configs/_template.yaml`)

**Files:**
- Create: `src/fine_tune/configs/_template.yaml`

- [ ] **Step 1: Write the template**

Paste verbatim (matches spec §4.1):

```yaml
# ============================================================
# Fine-tune config template — copy & adjust per model.
# Field-by-field rationale inline.
# ============================================================

model:
  id: Qwen/Qwen3.5-9B                    # HuggingFace model ID of the base VLM to fine-tune
  trust_remote_code: true                # required for some VLM custom classes (Qwen VL, Gemma 3/4)
  torch_dtype: bfloat16                  # bf16 = half memory of fp32, no loss scaling needed (vs fp16)
  attn_implementation: flash_attention_2 # FA2 is 2-3x faster than sdpa on Ampere+; falls back if unavailable

lora:
  r: 64                                  # rank of LoRA decomposition; 64 is Skin-R1/DermoGPT standard
  alpha: 128                             # LoRA scaling factor; alpha=2*r is the PEFT rule of thumb
  dropout: 0.05                          # regularization on LoRA updates; 0.05 is standard for SFT
  bias: none                             # don't train bias terms; saves params with ~no accuracy cost
  target_modules:                        # which linear layers get LoRA adapters (names from named_modules)
    - q_proj                             # query projection in attention
    - k_proj                             # key projection in attention
    - v_proj                             # value projection in attention
    - o_proj                             # output projection after attention
    - gate_proj                          # gating projection in MLP (SwiGLU)
    - up_proj                            # up-projection in MLP (expansion)
    - down_proj                          # down-projection in MLP (contraction)
  modules_to_save: []                    # extra modules to train in full (e.g., "lm_head") — empty = LoRA-only
  freeze_vision_tower: true              # keep MedSigLIP / Qwen ViT frozen; train only language-side

training:
  output_dir: /workspace/runs/qwen3.5-9b # where checkpoints + logs land on the volume

  # --- Duration ---
  num_train_epochs: 5                    # 5 passes over 28k samples; cosine LR decays to ~0 by the end

  # --- Batch math ---
  per_device_train_batch_size: 1         # samples per GPU step (kept low for 9B + image tokens)
  gradient_accumulation_steps: 16        # steps before optimizer.step(); effective batch = 1 * 16 = 16

  # --- Optimizer ---
  learning_rate: 1.0e-4                  # peak LR; 1e-4 is standard PEFT/LoRA for VLMs
  warmup_ratio: 0.03                     # fraction of total steps for linear warmup (stabilizes early training)
  lr_scheduler_type: cosine              # cosine decay from peak to ~0; smooth landing, no rude drop
  optim: paged_adamw_8bit                # 8-bit AdamW with pageable states; saves ~30% memory (9B only)
  weight_decay: 0.01                     # mild L2 regularization on non-LoRA weights (mostly no-op here)

  # --- Precision + memory ---
  bf16: true                             # compute in bfloat16; must match torch_dtype
  gradient_checkpointing: true           # trades recompute for memory; ~30% slower, ~40% less VRAM

  # --- Logging / eval / save ---
  logging_steps: 10                      # log loss every N steps; shows training curve live
  eval_strategy: steps                   # run val loop at step intervals (not per-epoch)
  eval_steps: 200                        # every 200 steps (~once per 10-15 min on 9B)
  save_strategy: epoch                   # checkpoint at end of each epoch
  save_total_limit: 3                    # keep only the 3 most recent epoch checkpoints to save disk
  load_best_model_at_end: true           # after training, reload the epoch with lowest val loss
  metric_for_best_model: eval_loss       # criterion for "best checkpoint"
  greater_is_better: false               # lower eval_loss is better

  # --- Reproducibility ---
  seed: 42                               # same seed across models = identical data shuffle

  # --- Monitoring ---
  report_to: [tensorboard]               # TB event files in output_dir/runs; csv log also on volume

data:
  train_file: /workspace/data/reasoning/train.jsonl  # prepared jsonl with absolute image paths
  val_file:   /workspace/data/reasoning/val.jsonl    # 5% stratified holdout
  max_seq_length: 4096                   # hard cap on tokenized sample length; truncates if over
  image_max_pixels: 1048576              # resize guard (≈1024x1024) so image tokens don't explode

hub:
  repo_id: danielfdias98/qwen3.5-9b-derm-reasoning  # HF Hub repo to push final adapter to
  push_strategy: end                     # push only the best checkpoint at run end (not per-epoch)
  private: true                          # keep unpublished until dissertation-ready
```

- [ ] **Step 2: Commit**

```bash
git add src/fine_tune/configs/_template.yaml
git commit -m "docs(fine_tune): annotated config template"
```

---

## Task 4: Four per-model YAML configs

**Files:**
- Create: `src/fine_tune/configs/medgemma-1.5-4b.yaml`
- Create: `src/fine_tune/configs/gemma-4-e4b.yaml`
- Create: `src/fine_tune/configs/qwen3.5-4b.yaml`
- Create: `src/fine_tune/configs/qwen3.5-9b.yaml`

- [ ] **Step 1: Write `medgemma-1.5-4b.yaml`**

```yaml
model:
  id: google/medgemma-1.5-4b-it
  trust_remote_code: true
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  bias: none
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  modules_to_save: []
  freeze_vision_tower: true

training:
  output_dir: /workspace/runs/medgemma-1.5-4b
  num_train_epochs: 5
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  optim: adamw_torch
  weight_decay: 0.01
  bf16: true
  gradient_checkpointing: true
  logging_steps: 10
  eval_strategy: steps
  eval_steps: 200
  save_strategy: epoch
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  seed: 42
  report_to: [tensorboard]

data:
  train_file: /workspace/data/reasoning/train.jsonl
  val_file:   /workspace/data/reasoning/val.jsonl
  max_seq_length: 4096
  image_max_pixels: 1048576

hub:
  repo_id: danielfdias98/medgemma-1.5-4b-derm-reasoning
  push_strategy: end
  private: true
```

- [ ] **Step 2: Write `gemma-4-e4b.yaml`**

Identical to the MedGemma config except:

```yaml
model:
  id: google/gemma-4-E4B-it
# ...rest same as medgemma-1.5-4b.yaml except...
training:
  output_dir: /workspace/runs/gemma-4-e4b
# ...
hub:
  repo_id: danielfdias98/gemma-4-e4b-derm-reasoning
```

Produce the full file by copying `medgemma-1.5-4b.yaml` and updating those three lines.

- [ ] **Step 3: Write `qwen3.5-4b.yaml`**

Full file:

```yaml
model:
  id: Qwen/Qwen3.5-4B
  trust_remote_code: true
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  bias: none
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  modules_to_save: []
  freeze_vision_tower: true

training:
  output_dir: /workspace/runs/qwen3.5-4b
  num_train_epochs: 5
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  optim: adamw_torch
  weight_decay: 0.01
  bf16: true
  gradient_checkpointing: true
  logging_steps: 10
  eval_strategy: steps
  eval_steps: 200
  save_strategy: epoch
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  seed: 42
  report_to: [tensorboard]

data:
  train_file: /workspace/data/reasoning/train.jsonl
  val_file:   /workspace/data/reasoning/val.jsonl
  max_seq_length: 4096
  image_max_pixels: 1048576

hub:
  repo_id: danielfdias98/qwen3.5-4b-derm-reasoning
  push_strategy: end
  private: true
```

- [ ] **Step 4: Write `qwen3.5-9b.yaml`**

Full file:

```yaml
model:
  id: Qwen/Qwen3.5-9B
  trust_remote_code: true
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  bias: none
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  modules_to_save: []
  freeze_vision_tower: true

training:
  output_dir: /workspace/runs/qwen3.5-9b
  num_train_epochs: 5
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  optim: paged_adamw_8bit
  weight_decay: 0.01
  bf16: true
  gradient_checkpointing: true
  logging_steps: 10
  eval_strategy: steps
  eval_steps: 200
  save_strategy: epoch
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  seed: 42
  report_to: [tensorboard]

data:
  train_file: /workspace/data/reasoning/train.jsonl
  val_file:   /workspace/data/reasoning/val.jsonl
  max_seq_length: 4096
  image_max_pixels: 1048576

hub:
  repo_id: danielfdias98/qwen3.5-9b-derm-reasoning
  push_strategy: end
  private: true
```

- [ ] **Step 5: Commit**

```bash
git add src/fine_tune/configs/medgemma-1.5-4b.yaml src/fine_tune/configs/gemma-4-e4b.yaml src/fine_tune/configs/qwen3.5-4b.yaml src/fine_tune/configs/qwen3.5-9b.yaml
git commit -m "feat(fine_tune): per-model YAML configs (4 models)"
```

---

## Task 5: Config loader (`config.py`)

**Files:**
- Create: `src/fine_tune/config.py`
- Create: `tests/fine_tune/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for src/fine_tune/config.py."""
from pathlib import Path

import pytest

from fine_tune.config import RunConfig, load_config

CONFIG_PATH = Path("src/fine_tune/configs/qwen3.5-9b.yaml")


def test_load_config_returns_run_config():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, RunConfig)


def test_load_config_parses_model_section():
    cfg = load_config(CONFIG_PATH)
    assert cfg.model.id == "Qwen/Qwen3.5-9B"
    assert cfg.model.torch_dtype == "bfloat16"
    assert cfg.model.trust_remote_code is True


def test_load_config_parses_lora_section():
    cfg = load_config(CONFIG_PATH)
    assert cfg.lora.r == 64
    assert cfg.lora.alpha == 128
    assert "q_proj" in cfg.lora.target_modules
    assert cfg.lora.freeze_vision_tower is True


def test_load_config_parses_training_section():
    cfg = load_config(CONFIG_PATH)
    assert cfg.training.num_train_epochs == 5
    assert cfg.training.per_device_train_batch_size == 1
    assert cfg.training.gradient_accumulation_steps == 16
    assert cfg.training.optim == "paged_adamw_8bit"
    assert cfg.training.learning_rate == pytest.approx(1e-4)


def test_load_config_parses_hub_section():
    cfg = load_config(CONFIG_PATH)
    assert cfg.hub.repo_id == "danielfdias98/qwen3.5-9b-derm-reasoning"
    assert cfg.hub.private is True


def test_load_all_four_configs():
    """Sanity: all four checked-in YAMLs parse into RunConfig."""
    for name in ("medgemma-1.5-4b", "gemma-4-e4b", "qwen3.5-4b", "qwen3.5-9b"):
        cfg = load_config(Path(f"src/fine_tune/configs/{name}.yaml"))
        assert cfg.model.id
        assert cfg.hub.repo_id.endswith("-derm-reasoning")
```

- [ ] **Step 2: Run test to verify failure**

Run: `PYTHONPATH=src pytest tests/fine_tune/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fine_tune.config'`

- [ ] **Step 3: Implement `src/fine_tune/config.py`**

```python
"""Typed config loader. YAML -> RunConfig dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    id: str
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"


@dataclass
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    bias: str
    target_modules: list[str]
    modules_to_save: list[str] = field(default_factory=list)
    freeze_vision_tower: bool = True


@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    lr_scheduler_type: str
    optim: str
    weight_decay: float
    bf16: bool
    gradient_checkpointing: bool
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    seed: int
    report_to: list[str]


@dataclass
class DataConfig:
    train_file: str
    val_file: str
    max_seq_length: int
    image_max_pixels: int


@dataclass
class HubConfig:
    repo_id: str
    push_strategy: str = "end"
    private: bool = True


@dataclass
class RunConfig:
    model: ModelConfig
    lora: LoraConfig
    training: TrainingConfig
    data: DataConfig
    hub: HubConfig
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return RunConfig(
        model=ModelConfig(**raw["model"]),
        lora=LoraConfig(**raw["lora"]),
        training=TrainingConfig(**raw["training"]),
        data=DataConfig(**raw["data"]),
        hub=HubConfig(**raw["hub"]),
        raw=raw,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/fine_tune/test_config.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fine_tune/config.py tests/fine_tune/test_config.py
git commit -m "feat(fine_tune): typed YAML config loader"
```

---

## Task 6: `data.py` — dataset + collator with loss masking

The collator's `labels = -100` on user-turn tokens is the only non-trivial logic. Tested with a mock processor.

**Files:**
- Create: `src/fine_tune/data.py`
- Create: `tests/fine_tune/test_data.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for src/fine_tune/data.py."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import torch

from fine_tune.data import MultimodalCollator, load_chat_dataset


def make_chat_row(image_path: str, answer_json: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Diagnose this skin condition with structured reasoning."},
            ]},
            {"role": "assistant", "content": answer_json},
        ]
    }


def test_load_chat_dataset_reads_jsonl(tmp_path: Path):
    jsonl = tmp_path / "tiny.jsonl"
    rows = [make_chat_row("/fake/a.jpg", '{"diagnosis":"A"}'),
            make_chat_row("/fake/b.jpg", '{"diagnosis":"B"}')]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    ds = load_chat_dataset(jsonl)
    assert len(ds) == 2
    assert ds[0]["messages"][0]["role"] == "user"


class FakeProcessor:
    """Minimal processor double that mimics apply_chat_template + image handling."""
    def __init__(self):
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0

    def apply_chat_template(self, conversation, tokenize=True, return_dict=True,
                             add_generation_prompt=False, **kwargs):
        # Return deterministic ids based on role counts: user=[1,2,3], assistant=[4,5]
        user_turns = [m for m in conversation if m["role"] == "user"]
        assistant_turns = [m for m in conversation if m["role"] == "assistant"]
        ids = []
        for _ in user_turns:
            ids.extend([1, 2, 3])
        for _ in assistant_turns:
            ids.extend([4, 5])
        input_ids = torch.tensor([ids])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.zeros(1, 3, 8, 8),
        }


def test_collator_masks_user_tokens_in_labels(monkeypatch, tmp_path: Path):
    # Patch PIL.Image.open in the data module to avoid reading real files
    from fine_tune import data as data_mod
    monkeypatch.setattr(data_mod, "_open_image", lambda p: MagicMock(name=f"img:{p}"))

    processor = FakeProcessor()
    collator = MultimodalCollator(processor=processor, max_seq_length=2048)

    row = make_chat_row("/fake/a.jpg", '{"diagnosis":"Melanoma"}')
    batch = collator([row])

    # input_ids should be [1,2,3,4,5] -> first 3 are user (masked), last 2 are assistant (kept)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5]]


def test_collator_pads_batch_to_longest(monkeypatch):
    from fine_tune import data as data_mod
    monkeypatch.setattr(data_mod, "_open_image", lambda p: MagicMock())

    class VarProcessor(FakeProcessor):
        def __init__(self):
            super().__init__()
            self._call = 0

        def apply_chat_template(self, conversation, **kwargs):
            self._call += 1
            ids = [1, 2, 3] if self._call == 1 else [1, 2, 3, 4, 5, 6]
            input_ids = torch.tensor([ids])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "pixel_values": torch.zeros(1, 3, 8, 8),
            }

    collator = MultimodalCollator(processor=VarProcessor(), max_seq_length=2048)
    rows = [make_chat_row("/fake/a.jpg", "{}"), make_chat_row("/fake/b.jpg", "{}")]
    batch = collator(rows)
    assert batch["input_ids"].shape == (2, 6)  # padded to longest
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/fine_tune/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fine_tune.data'`

- [ ] **Step 3: Implement `src/fine_tune/data.py`**

```python
"""Dataset + collator for VLM SFT.

The collator:
  1. Loads each row's image from disk
  2. Applies the processor's chat template to produce input_ids + pixel_values
  3. Masks user-turn tokens in `labels` (-100) so loss is computed only on assistant turn
  4. Pads the batch to the longest sample
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image


def _open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_chat_dataset(path: Path) -> list[dict[str, Any]]:
    """Eagerly load a chat-format jsonl into a list of dicts.

    28k rows × small dict easily fits RAM; images stay on disk and are
    read per-batch inside the collator.
    """
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


class MultimodalCollator:
    """Collate a list of chat rows into a padded batch with masked labels."""

    def __init__(self, processor: Any, max_seq_length: int):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.pad_token_id = getattr(getattr(processor, "tokenizer", processor), "pad_token_id", 0) or 0

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids_list: list[torch.Tensor] = []
        label_list: list[torch.Tensor] = []
        pixel_list: list[torch.Tensor] = []

        for row in rows:
            conversation, images = self._split_row(row)

            # Full conversation: user + assistant
            full = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=False,
                images=images if images else None,
            )
            full_ids = full["input_ids"].squeeze(0)

            # User-only prefix: same turns minus assistant
            user_only = [m for m in conversation if m["role"] == "user"]
            prefix = self.processor.apply_chat_template(
                user_only,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                images=images if images else None,
            )
            prefix_len = prefix["input_ids"].shape[-1]

            labels = full_ids.clone()
            labels[:prefix_len] = -100

            input_ids_list.append(full_ids)
            label_list.append(labels)
            if "pixel_values" in full:
                pixel_list.append(full["pixel_values"].squeeze(0) if full["pixel_values"].dim() == 4 else full["pixel_values"])

        input_ids, attention_mask, labels = self._pad(input_ids_list, label_list)
        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_list:
            batch["pixel_values"] = torch.stack(pixel_list, dim=0) if pixel_list[0].dim() == 3 else torch.cat(pixel_list, dim=0)
        return batch

    def _split_row(self, row: dict[str, Any]) -> tuple[list[dict], list[Image.Image]]:
        messages = row["messages"]
        images: list[Image.Image] = []
        clean_messages: list[dict] = []
        for m in messages:
            if isinstance(m["content"], list):
                parts: list[dict] = []
                for c in m["content"]:
                    if c.get("type") == "image":
                        images.append(_open_image(c["image"]))
                        parts.append({"type": "image"})
                    else:
                        parts.append(c)
                clean_messages.append({"role": m["role"], "content": parts})
            else:
                clean_messages.append(m)
        return clean_messages, images

    def _pad(self, ids: list[torch.Tensor], labels: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = min(self.max_seq_length, max(x.shape[-1] for x in ids))
        b = len(ids)
        out_ids = torch.full((b, max_len), self.pad_token_id, dtype=ids[0].dtype)
        out_mask = torch.zeros((b, max_len), dtype=torch.long)
        out_labels = torch.full((b, max_len), -100, dtype=labels[0].dtype)
        for i, (x, y) in enumerate(zip(ids, labels)):
            n = min(x.shape[-1], max_len)
            out_ids[i, :n] = x[:n]
            out_mask[i, :n] = 1
            out_labels[i, :n] = y[:n]
        return out_ids, out_mask, out_labels
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/fine_tune/test_data.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fine_tune/data.py tests/fine_tune/test_data.py
git commit -m "feat(fine_tune): dataset loader + multimodal collator with label masking"
```

---

## Task 7: `model_loader.py` — HF model + processor + LoRA + vision freeze

**Files:**
- Create: `src/fine_tune/model_loader.py`
- Create: `tests/fine_tune/test_model_loader.py`

- [ ] **Step 1: Write tests for the pure-logic helper**

```python
"""Tests for the vision-tower freeze logic (the only non-HF piece of model_loader)."""
from unittest.mock import MagicMock

import torch

from fine_tune.model_loader import freeze_vision_tower


def make_fake_model_with_params(names: list[str]):
    class Fake:
        def __init__(self):
            self._params = {n: torch.nn.Parameter(torch.zeros(1)) for n in names}

        def named_parameters(self):
            return self._params.items()

    return Fake()


def test_freeze_vision_tower_disables_grad_on_matching_names():
    model = make_fake_model_with_params([
        "vision_tower.encoder.layers.0.weight",
        "vision_model.embeddings.patch_embedding.weight",
        "visual.blocks.0.attn.weight",
        "language_model.layers.0.self_attn.q_proj.weight",
    ])
    freeze_vision_tower(model)
    for name, p in model.named_parameters():
        if "vision_tower" in name or "vision_model" in name or "visual" in name:
            assert not p.requires_grad, f"expected {name} frozen"
        else:
            assert p.requires_grad, f"expected {name} trainable"
```

- [ ] **Step 2: Run test to verify failure**

Run: `PYTHONPATH=src pytest tests/fine_tune/test_model_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/fine_tune/model_loader.py`**

```python
"""Load base VLM + processor + apply LoRA."""
from __future__ import annotations

from typing import Any

import torch

from fine_tune.config import LoraConfig, ModelConfig


_VISION_PREFIXES = ("vision_tower", "vision_model", "visual")


def freeze_vision_tower(model: Any) -> int:
    """Set requires_grad=False on params whose names start with a vision prefix.

    Returns the number of params frozen.
    """
    count = 0
    for name, p in model.named_parameters():
        if any(prefix in name for prefix in _VISION_PREFIXES):
            p.requires_grad = False
            count += 1
    return count


def _dtype_from_str(s: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[s]


def load_model_and_processor(model_cfg: ModelConfig, lora_cfg: LoraConfig):
    """Load base model + processor, wrap with PEFT LoRA, optionally freeze vision.

    Import HF libs lazily so tests that only touch `freeze_vision_tower` don't
    need transformers/peft installed.
    """
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModelForVision2Seq,
        AutoProcessor,
    )
    from peft import LoraConfig as PeftLoraConfig, get_peft_model

    torch_dtype = _dtype_from_str(model_cfg.torch_dtype)

    # Dispatch to the right AutoModel class
    config = AutoConfig.from_pretrained(model_cfg.id, trust_remote_code=model_cfg.trust_remote_code)
    arch = (config.architectures or ["AutoModelForCausalLM"])[0]

    if "Vision2Seq" in arch or "Qwen" in arch and "VL" in arch:
        loader = AutoModelForVision2Seq
    elif "ImageTextToText" in arch or arch.startswith("Gemma3") or arch.startswith("MedGemma"):
        loader = AutoModelForImageTextToText
    else:
        loader = AutoModelForCausalLM

    model = loader.from_pretrained(
        model_cfg.id,
        torch_dtype=torch_dtype,
        attn_implementation=model_cfg.attn_implementation,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(model_cfg.id, trust_remote_code=model_cfg.trust_remote_code)

    if lora_cfg.freeze_vision_tower:
        freeze_vision_tower(model)

    peft_cfg = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        target_modules=lora_cfg.target_modules,
        modules_to_save=lora_cfg.modules_to_save or None,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, processor
```

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=src pytest tests/fine_tune/test_model_loader.py -v`
Expected: 1 test PASS. (The HF loading path is GPU-verified later, not unit-tested.)

- [ ] **Step 5: Commit**

```bash
git add src/fine_tune/model_loader.py tests/fine_tune/test_model_loader.py
git commit -m "feat(fine_tune): model_loader with LoRA + vision-tower freeze"
```

---

## Task 8: `callbacks.py` — CSV logger + HF Hub push

**Files:**
- Create: `src/fine_tune/callbacks.py`
- Create: `tests/fine_tune/test_callbacks.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for src/fine_tune/callbacks.py."""
import csv
from pathlib import Path
from unittest.mock import MagicMock

from fine_tune.callbacks import CSVLoggerCallback


def test_csv_logger_writes_header_and_row(tmp_path: Path):
    out = tmp_path / "metrics.csv"
    cb = CSVLoggerCallback(output_path=out)
    args = MagicMock()
    state = MagicMock(global_step=10, epoch=0.5)
    control = MagicMock()
    cb.on_log(args, state, control, logs={"loss": 1.23, "learning_rate": 1e-4})
    rows = list(csv.reader(out.read_text().splitlines()))
    assert rows[0] == ["step", "epoch", "loss", "eval_loss", "learning_rate"]
    assert rows[1][0] == "10"
    assert rows[1][2] == "1.23"


def test_csv_logger_appends_row_when_file_exists(tmp_path: Path):
    out = tmp_path / "metrics.csv"
    cb = CSVLoggerCallback(output_path=out)
    args = MagicMock()
    state = MagicMock(global_step=10, epoch=0.5)
    cb.on_log(args, state, MagicMock(), logs={"loss": 1.0})
    cb.on_log(args, MagicMock(global_step=20, epoch=1.0), MagicMock(), logs={"loss": 0.5})
    lines = out.read_text().splitlines()
    assert len(lines) == 3  # header + 2 rows
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src pytest tests/fine_tune/test_callbacks.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/fine_tune/callbacks.py`**

```python
"""TrainerCallbacks for CSV logging and HF Hub push."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class CSVLoggerCallback:
    """Writes step-level metrics to a CSV on the volume (parseable for dissertation tables)."""

    HEADER = ["step", "epoch", "loss", "eval_loss", "learning_rate"]

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._written_header = self.output_path.exists()

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs) -> None:
        if not logs:
            return
        row = [
            str(state.global_step),
            f"{state.epoch:.4f}" if state.epoch is not None else "",
            str(logs.get("loss", "")),
            str(logs.get("eval_loss", "")),
            str(logs.get("learning_rate", "")),
        ]
        new_file = not self._written_header
        with self.output_path.open("a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(self.HEADER)
                self._written_header = True
            w.writerow(row)


class HFHubPushCallback:
    """At on_train_end, push the (best, already-loaded) PEFT adapter to HF Hub.

    Uploads only the adapter (~100-400 MB), not the base model weights.
    """

    def __init__(self, repo_id: str, private: bool, config_yaml_path: Path):
        self.repo_id = repo_id
        self.private = private
        self.config_yaml_path = Path(config_yaml_path)

    def on_train_end(self, args: Any, state: Any, control: Any, model=None, **kwargs) -> None:
        if model is None:
            return
        best_epoch = _best_epoch_from_state(state)
        best_loss = state.best_metric if getattr(state, "best_metric", None) is not None else "n/a"
        commit_msg = f"Fine-tuned on 28k structured reasoning — best epoch {best_epoch}, eval_loss {best_loss}"
        model.push_to_hub(self.repo_id, private=self.private, commit_message=commit_msg)
        self._upload_config(commit_msg)

    def _upload_config(self, commit_msg: str) -> None:
        from huggingface_hub import upload_file
        if not self.config_yaml_path.exists():
            return
        upload_file(
            path_or_fileobj=str(self.config_yaml_path),
            path_in_repo="config.yaml",
            repo_id=self.repo_id,
            commit_message=f"Add training config — {commit_msg}",
        )


def _best_epoch_from_state(state: Any) -> str:
    history = getattr(state, "log_history", []) or []
    best = None
    for row in history:
        if "eval_loss" in row:
            if best is None or row["eval_loss"] < best[1]:
                best = (row.get("epoch", "?"), row["eval_loss"])
    return str(best[0]) if best else "?"
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/fine_tune/test_callbacks.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fine_tune/callbacks.py tests/fine_tune/test_callbacks.py
git commit -m "feat(fine_tune): CSV logger + HF Hub push callbacks"
```

---

## Task 9: `train.py` — orchestrator with `--dry-run` mode

**Files:**
- Create: `src/fine_tune/train.py`

No local unit tests — verified via `--dry-run` on the pod (Task 14).

- [ ] **Step 1: Write `src/fine_tune/train.py`**

```python
"""Fine-tune a VLM using TRL SFTTrainer + PEFT LoRA.

Usage:
    python -m fine_tune.train --config src/fine_tune/configs/qwen3.5-9b.yaml
    python -m fine_tune.train --config ... --resume             # resume from latest checkpoint
    python -m fine_tune.train --config ... --dry-run            # one fwd/bwd step to detect OOM
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import TrainingArguments
from transformers.integrations import TensorBoardCallback
from trl import SFTTrainer

from fine_tune.callbacks import CSVLoggerCallback, HFHubPushCallback
from fine_tune.config import RunConfig, load_config
from fine_tune.data import MultimodalCollator, load_chat_dataset
from fine_tune.model_loader import load_model_and_processor


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _set_seeds(seed: int) -> None:
    import random, numpy as np, transformers
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    transformers.set_seed(seed)


def _copy_config_into_output_dir(cfg_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "config.yaml"
    shutil.copy(cfg_path, dest)
    return dest


def _build_training_args(cfg: RunConfig) -> TrainingArguments:
    t = cfg.training
    return TrainingArguments(
        output_dir=t.output_dir,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        optim=t.optim,
        weight_decay=t.weight_decay,
        bf16=t.bf16,
        gradient_checkpointing=t.gradient_checkpointing,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        seed=t.seed,
        report_to=t.report_to,
        remove_unused_columns=False,
    )


def _append_manifest_row(cfg: RunConfig, cfg_sha: str, wall_clock_hours: float, best_loss: float | None) -> None:
    manifest = Path("/workspace/runs/manifest.csv")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    is_new = not manifest.exists()
    with manifest.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "model", "config_sha", "best_eval_loss", "wall_clock_hours", "hub_repo"])
        w.writerow([
            datetime.utcnow().isoformat(timespec="seconds"),
            cfg.model.id,
            cfg_sha,
            f"{best_loss:.4f}" if best_loss is not None else "n/a",
            f"{wall_clock_hours:.2f}",
            cfg.hub.repo_id,
        ])


def _dry_run(trainer: SFTTrainer) -> None:
    """One forward+backward step to detect OOM / wiring errors before committing to an epoch."""
    print("[dry-run] Fetching one batch + executing one training step…")
    loader = trainer.get_train_dataloader()
    batch = next(iter(loader))
    trainer.model.train()
    outputs = trainer.model(**{k: v.to(trainer.model.device) for k, v in batch.items() if torch.is_tensor(v)})
    loss = outputs.loss
    loss.backward()
    print(f"[dry-run] OK. loss={loss.item():.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output_dir")
    p.add_argument("--dry-run", action="store_true", help="One fwd/bwd step, no real training")
    args = p.parse_args()

    cfg = load_config(args.config)
    _set_seeds(cfg.training.seed)

    output_dir = Path(cfg.training.output_dir)
    saved_cfg_path = _copy_config_into_output_dir(args.config, output_dir)

    print(f"Loading base model + processor: {cfg.model.id}")
    model, processor = load_model_and_processor(cfg.model, cfg.lora)

    print(f"Loading data: train={cfg.data.train_file} val={cfg.data.val_file}")
    train_ds = load_chat_dataset(Path(cfg.data.train_file))
    val_ds = load_chat_dataset(Path(cfg.data.val_file))

    collator = MultimodalCollator(processor=processor, max_seq_length=cfg.data.max_seq_length)

    training_args = _build_training_args(cfg)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=getattr(processor, "tokenizer", processor),
    )

    trainer.add_callback(CSVLoggerCallback(output_path=output_dir / "metrics.csv"))
    trainer.add_callback(HFHubPushCallback(
        repo_id=cfg.hub.repo_id,
        private=cfg.hub.private,
        config_yaml_path=saved_cfg_path,
    ))
    trainer.add_callback(TensorBoardCallback())

    if args.dry_run:
        _dry_run(trainer)
        return

    start = time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    wall_clock_hours = (time.time() - start) / 3600

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))

    best_loss = trainer.state.best_metric
    _append_manifest_row(cfg, _git_sha(), wall_clock_hours, best_loss)

    print(f"Training complete. Best eval_loss={best_loss}, wall={wall_clock_hours:.2f}h")
    print(f"Final adapter: {final_dir}")
    print(f"HF Hub repo:   https://huggingface.co/{cfg.hub.repo_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/fine_tune/train.py
git commit -m "feat(fine_tune): training orchestrator with dry-run + manifest ledger"
```

---

## Task 10: RunPod shell scripts

**Files:**
- Create: `src/fine_tune/scripts/sync_to_volume.sh`
- Create: `src/fine_tune/scripts/setup_pod.sh`
- Create: `src/fine_tune/scripts/run.sh`

- [ ] **Step 1: Write `sync_to_volume.sh`**

```bash
#!/usr/bin/env bash
# Upload final/train/ (29,913 images) + data/reasoning/reasoning.jsonl to the RunPod volume.
# Run from laptop. Idempotent (rsync skips unchanged).
#
# Usage:
#   POD_ALIAS=runpod-train ./src/fine_tune/scripts/sync_to_volume.sh
set -euo pipefail

POD_ALIAS="${POD_ALIAS:-runpod-train}"
REMOTE_ROOT="/workspace/data"

if [ ! -d "final/train" ]; then
  echo "ERROR: ./final/train not found. Run from repo root."; exit 1
fi
if [ ! -f "data/reasoning/reasoning.jsonl" ]; then
  echo "ERROR: ./data/reasoning/reasoning.jsonl not found."; exit 1
fi

echo "Syncing final/train/ → ${POD_ALIAS}:${REMOTE_ROOT}/images/final/train/"
rsync -avz --progress --partial \
  final/train/ \
  "${POD_ALIAS}:${REMOTE_ROOT}/images/final/train/"

echo "Syncing reasoning.jsonl → ${POD_ALIAS}:${REMOTE_ROOT}/reasoning/"
ssh "${POD_ALIAS}" "mkdir -p ${REMOTE_ROOT}/reasoning"
rsync -avz --progress \
  data/reasoning/reasoning.jsonl \
  "${POD_ALIAS}:${REMOTE_ROOT}/reasoning/reasoning.jsonl"

echo "Sync complete. On pod, run: python -m fine_tune.prepare_data --image-root /workspace/data/images --source /workspace/data/reasoning/reasoning.jsonl --train-out /workspace/data/reasoning/train.jsonl --val-out /workspace/data/reasoning/val.jsonl --audit-out /workspace/data/reasoning/audit.jsonl"
```

- [ ] **Step 2: Write `setup_pod.sh`**

```bash
#!/usr/bin/env bash
# Install CUDA deps, HF login, verify GPU + data visibility.
# Run once per fresh pod. HF_TOKEN must be set in env beforehand.
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p /workspace/hf_cache /workspace/runs

REQS="$(dirname "$0")/../requirements-gpu.txt"
pip install --upgrade -r "${REQS}"
pip install --upgrade 'flash-attn>=2.6' --no-build-isolation

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN env var not set. export HF_TOKEN=hf_... and re-run."; exit 1
fi
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential

python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK. GPUs: {torch.cuda.device_count()}')"
ls /workspace/data/reasoning/reasoning.jsonl >/dev/null && echo "Data OK: reasoning.jsonl"
ls /workspace/data/images/final/train >/dev/null && echo "Data OK: images/final/train"
echo "Pod setup complete."
```

- [ ] **Step 3: Write `run.sh`**

```bash
#!/usr/bin/env bash
# Usage: ./scripts/run.sh src/fine_tune/configs/qwen3.5-9b.yaml [--resume|--dry-run]
set -euo pipefail

CONFIG="${1:?usage: run.sh <config> [--resume|--dry-run]}"
shift || true

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python -m fine_tune.train --config "${CONFIG}" "$@"
```

- [ ] **Step 4: Make scripts executable and commit**

```bash
chmod +x src/fine_tune/scripts/*.sh
git add src/fine_tune/scripts/
git update-index --chmod=+x src/fine_tune/scripts/sync_to_volume.sh src/fine_tune/scripts/setup_pod.sh src/fine_tune/scripts/run.sh
git commit -m "feat(fine_tune): RunPod shell scripts (sync, setup, run)"
```

---

## Task 11: Patch `src/eval/score_results.py` for `top_n` support

**Files:**
- Modify: `src/eval/score_results.py` (one location in `score_fitzpatrick`)

- [ ] **Step 1: Apply the patch**

Open `src/eval/score_results.py` and change the single line that reads `top_6`:

```python
# BEFORE:
top6 = [normalize_diagnosis(d) for d in parsed.get("top_6", [])]

# AFTER:
top_n_raw = parsed.get("top_n") or parsed.get("top_6") or []
top6 = [normalize_diagnosis(d) for d in top_n_raw]
```

- [ ] **Step 2: Sanity check — existing results still score the same**

Run on an existing results file (that uses `top_6`):
```bash
python src/eval/score_results.py --model medgemma-4b 2>&1 | head -20
```

Expected: same Top-1/Top-6 numbers as before (19.7% Top-1 / 31.1% Top-6 for MedGemma 4B, per `implementation_log.md` §7.10).

- [ ] **Step 3: Commit**

```bash
git add src/eval/score_results.py
git commit -m "feat(eval): accept top_n alongside top_6 in scorer"
```

---

## Task 12: Full local test pass + documentation update

**Files:**
- Modify: `docs/src/implementation_log.md` (append Phase 8 section)

- [ ] **Step 1: Run the full local test suite**

Run: `PYTHONPATH=src pytest tests/fine_tune/ -v`
Expected: all tests from Tasks 2, 5, 6, 7, 8 pass (~22 tests total).

- [ ] **Step 2: Append a Phase 8 section to `implementation_log.md`**

Add at the end of `docs/src/implementation_log.md`, before `## Next Steps`:

```markdown
## Phase 8: Fine-Tuning Pipeline (Implementation)

### 8.1 Pipeline Overview

`src/fine_tune/` implements a config-driven LoRA SFT pipeline over TRL + PEFT. One entrypoint (`python -m fine_tune.train --config <yaml>`) fine-tunes any of the four base VLMs; per-model variations live in four checked-in YAML configs. Vision towers are frozen (preserves MedSigLIP's medical advantage on MedGemma and Qwen's early-fusion features). LoRA rank 64 / α 128, 5 epochs, cosine LR schedule, `paged_adamw_8bit` on the 9B model only.

Data preparation runs once (`python -m fine_tune.prepare_data`): stratified 95/5 train/val split from the 28,486 reasoning entries, 139 flagged entries separated into an audit file and excluded from training. All four models train on byte-identical data. Deterministic with `seed=42`.

Training artifacts: per-epoch LoRA checkpoints on the `/workspace` network volume (save_total_limit=3), best-by-eval-loss adapter pushed to a private HF Hub repo (`danielfdias98/<model-name>-derm-reasoning`), and a `manifest.csv` ledger capturing one row per completed run (model, config SHA, best eval_loss, wall-clock hours).

### 8.2 Operational Workflow (RunPod L40S)

1. `scripts/sync_to_volume.sh` — one-time rsync of `final/train/` + `reasoning.jsonl` to the network volume (~30 min).
2. `scripts/setup_pod.sh` — once per fresh pod: installs GPU deps, HF login, verifies GPU + data.
3. `scripts/run.sh configs/<model>.yaml` — one invocation per model (~5–14 h each on L40S).

Pod restart mid-run: `scripts/run.sh configs/<model>.yaml --resume` auto-resumes from the latest epoch checkpoint.

### 8.3 Design & Plan References

- Design spec: `docs/superpowers/specs/2026-04-17-fine-tune-pipeline-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-17-fine-tune-pipeline.md`
```

- [ ] **Step 3: Commit**

```bash
git add docs/src/implementation_log.md
git commit -m "docs: record Phase 8 fine-tune pipeline in implementation_log"
```

---

## Task 13: GPU smoke test — `prepare_data` on real reasoning.jsonl

From here on, tasks run on a RunPod L40S pod with network volume attached and `HF_TOKEN` in env.

**Files:**
- None (verification only)

- [ ] **Step 1: SSH into pod, sync repo**

From laptop: `bash src/fine_tune/scripts/sync_to_volume.sh` (if not done).
Then SSH into the pod.

- [ ] **Step 2: Pull the repo on the pod**

```bash
cd /workspace
git clone https://github.com/danielferreira-dias/dissertation.git  # skip if cloned
cd dissertation && git pull
```

- [ ] **Step 3: Run setup**

```bash
export HF_TOKEN=hf_xxx                       # from HF settings → tokens
bash src/fine_tune/scripts/setup_pod.sh
```

Expected: `CUDA OK. GPUs: 1` and both `Data OK` lines printed.

- [ ] **Step 4: Run prepare_data on real data**

```bash
cd /workspace/dissertation
export PYTHONPATH=src
python -m fine_tune.prepare_data \
  --source /workspace/data/reasoning/reasoning.jsonl \
  --train-out /workspace/data/reasoning/train.jsonl \
  --val-out /workspace/data/reasoning/val.jsonl \
  --audit-out /workspace/data/reasoning/audit.jsonl \
  --image-root /workspace/data/images \
  --val-fraction 0.05 \
  --seed 42
```

Expected output:
```
Source entries:     28486
Audited (flagged):  139
Train examples:     ~26900
Val examples:       ~1447
Classes:            337
```

Sanity check: `head -1 /workspace/data/reasoning/train.jsonl | python -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['messages'][1]['content'][:200])"` — should print a JSON payload starting with `{"diagnosis":`.

---

## Task 14: GPU smoke test — `train.py --dry-run` on MedGemma 4B

The smallest model. Catches wiring / OOM issues before running a full epoch.

- [ ] **Step 1: Invoke dry-run**

```bash
cd /workspace/dissertation
export HF_HOME=/workspace/hf_cache
export PYTHONPATH=src
bash src/fine_tune/scripts/run.sh src/fine_tune/configs/medgemma-1.5-4b.yaml --dry-run
```

Expected:
- Base model downloads to `/workspace/hf_cache/` (~8 GB; once-per-model).
- PEFT prints trainable parameter count (~0.2–0.5 % of total).
- `[dry-run] OK. loss=<value>` printed.
- Exit code 0.

If OOM: reduce `per_device_train_batch_size` to 1 in the YAML, re-run.
If FA2 install failed: edit YAML `attn_implementation: sdpa`, re-run.

- [ ] **Step 2: Commit any YAML edits needed to make dry-run pass**

```bash
git add src/fine_tune/configs/
git commit -m "tune(fine_tune): YAML tweaks from pod dry-run on MedGemma 4B"
git push
```

---

## Task 15: Full training run — MedGemma 1.5 4B

The first real run. Sets a baseline for wall-clock estimates used in remaining runs.

- [ ] **Step 1: Start the run**

```bash
cd /workspace/dissertation
bash src/fine_tune/scripts/run.sh src/fine_tune/configs/medgemma-1.5-4b.yaml 2>&1 | tee /workspace/runs/medgemma-1.5-4b/run.log
```

Expected: training proceeds, `metrics.csv` grows, TB events appear in `/workspace/runs/medgemma-1.5-4b/runs/`. Estimated wall clock: 6–8 hours.

If pod interrupts: `bash src/fine_tune/scripts/run.sh src/fine_tune/configs/medgemma-1.5-4b.yaml --resume`.

- [ ] **Step 2: Verify artifacts after completion**

```bash
ls /workspace/runs/medgemma-1.5-4b/final/            # adapter_model.safetensors present
cat /workspace/runs/manifest.csv                      # new row with wall_clock + eval_loss
huggingface-cli repo view danielfdias98/medgemma-1.5-4b-derm-reasoning   # repo exists, private
```

- [ ] **Step 3: Smoke test adapter load**

```bash
python - <<'PY'
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
base = AutoModelForImageTextToText.from_pretrained("google/medgemma-1.5-4b-it", torch_dtype="bfloat16", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "danielfdias98/medgemma-1.5-4b-derm-reasoning")
print("Adapter loaded OK:", model.active_adapter)
PY
```

Expected: prints `Adapter loaded OK: default`.

- [ ] **Step 4: Kick off remaining three models (sequentially)**

Once the MedGemma run validates the pipeline, repeat for the other three configs. No code changes expected — only YAML swaps.

```bash
bash src/fine_tune/scripts/run.sh src/fine_tune/configs/qwen3.5-4b.yaml
bash src/fine_tune/scripts/run.sh src/fine_tune/configs/qwen3.5-9b.yaml
bash src/fine_tune/scripts/run.sh src/fine_tune/configs/gemma-4-e4b.yaml
```

Stop the pod between runs to halt billing.

---

## Completion Checklist

- [ ] All 4 LoRA adapters pushed to private HF Hub repos under `danielfdias98/*-derm-reasoning`.
- [ ] `/workspace/runs/manifest.csv` contains 4 rows, one per model.
- [ ] Each adapter loads cleanly via `PeftModel.from_pretrained(base, repo)`.
- [ ] `implementation_log.md` Phase 8 section committed.
- [ ] `src/eval/` run post-training produces valid JSON matching the new schema (`top_n` field populated).

Evaluation of fine-tuned models (running `src/eval/run_benchmark.py` against the adapters and comparing to zero-shot baselines) is out of scope for this plan — handled in a follow-up.
