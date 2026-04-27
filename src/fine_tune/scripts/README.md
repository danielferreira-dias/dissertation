# Fine-tune pod template

A reproducible environment for LoRA fine-tuning VLMs on a RunPod / Vast.ai / Lambda
GPU pod. Idempotent across pod restarts.

## What's here

- `setup_pod.sh` — `pip install unsloth + pure-Python deps`, log into HF, validate
- `verify_env.py` — sanity checks (torch CUDA, unsloth/trl/transformers/peft imports, env vars, bf16 matmul)
- `../requirements-gpu.txt` — `unsloth[cu124-torch260]` + non-bundled deps (PyYAML, tensorboard, sentencepiece, …)
- `sync_to_volume.sh` — rsync repo + image data from laptop to pod (legacy, unchanged)
- `run.sh` — kick off a single training run in tmux (legacy, unchanged)

The pipeline now uses **Unsloth's `FastVisionModel`** for model loading + LoRA, and **TRL's `SFTTrainer`** for the training loop (Unsloth bundles the matching torch/transformers/peft/accelerate/bitsandbytes/flash-attn/fla/triton stack as one install).

## Driver tier assumed

NVIDIA driver branch **R565** (`nvidia-smi` reports `CUDA Version: 12.7`). This is the
common RunPod tier as of April 2026. Supported toolkits: CUDA ≤ 12.6. Therefore:

- ✅ `torch+cu124` and `torch+cu126` wheels work
- ❌ `torch+cu128` and `torch+cu130` do NOT — `torch.cuda.is_available()` will be False

If your pod ships with a newer driver (R570+, reported as 12.8+), you can move to
`torch+cu126` or `cu128`, but you also have to bump kernel pins to match. Re-run
`verify_env.py` to confirm.

## First-time pod setup

```bash
# 1. SSH into a fresh pod
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

# 2. (Once) put your HF token on the persistent volume
echo "hf_…" > /workspace/.hf_token
chmod 600 /workspace/.hf_token

# 3. Clone the repo onto the persistent volume
cd /workspace && git clone https://github.com/danielferreira-dias/dissertation.git
cd dissertation

# 4. Install the pinned env
export HF_TOKEN=$(cat /workspace/.hf_token)
bash src/fine_tune/scripts/setup_pod.sh

# 5. Validate (already done by setup_pod.sh, but re-run anytime)
python src/fine_tune/scripts/verify_env.py
```

Expected verify_env.py output: 14 OK lines, no failures.

## After a pod restart

Container resets but `/workspace` survives. Re-run setup:

```bash
cd /workspace/dissertation
git pull
export HF_TOKEN=$(cat /workspace/.hf_token)
bash src/fine_tune/scripts/setup_pod.sh
```

If pip detects everything's already installed, it's a no-op except for env var setup.

## Why the pins are what they are

### Why torch 2.6.0+cu124 and not the latest

- Pod driver R565 supports CUDA toolkit ≤ 12.6 — `cu128`/`cu130` wheels fail at
  `torch.cuda.is_available() == False`.
- `torch+cu124` index has wheels up to 2.6.0 (last version in cu124 line). Past 2.6,
  PyTorch moved to cu126/128 only.
- Pre-built wheels exist for flash-attn, fla 0.4.x, causal-conv1d on torch 2.6 — no
  source compiles needed.

### Why flash-linear-attention 0.4.2 and not latest 0.5.0

- `fla 0.5.0` requires `fla-core 0.5.0` requires **torch ≥ 2.7**.
- torch 2.7+ ships only cu126/cu128 wheels — cu128 exceeds R565's CUDA capability.
- We hit this exactly: a `pip install -U flash-linear-attention` upgraded torch to
  2.11+cu130, broke `torch.cuda`, and cascaded through torchvision (ABI mismatch),
  transformers (depends on torchvision.io), peft (depends on transformers), and the
  CUDA kernels (compiled against the old torch ABI). One bad pip command, two hours
  of recovery.
- `fla 0.4.2` has no torch upper bound. Confirmed compatible with torch 2.4 / 2.5 / 2.6.

### Why transformers 5.6.2

- Required for Qwen 3.5 (`model_type: qwen3_5` is unknown to transformers 4.57.x).
- 5.6.2 is the latest stable; 5.7.0.dev0 (main) requires `torch ≥ 2.6` for new mask
  features and breaks MedGemma without warning.
- 4.x series doesn't support our newer models. Don't downgrade.

### Why peft 0.19.1 and not git main

- 0.19.1 is the latest pip release. Has Gemma 3 support but not yet Gemma 4 dispatch
  for `Gemma4ClippableLinear` (Gemma 4 is dropped from this campaign anyway).
- Git main is unstable; can break older models without warning.

## Adding a new model

For a model **not** in the existing 3 (MedGemma, Qwen 3.5 4B/9B), copy
`configs/medgemma-1.5-4b.yaml` to `configs/<new-model>.yaml` and edit:

1. `model.id` → HF Hub repo of the new model
2. `lora.target_modules` — verify the model uses standard `q_proj`/`k_proj`/etc names.
   If it has custom `Linear` wrappers (e.g. `Gemma4ClippableLinear`), use a regex
   scoped to `language_model`.
3. `training.per_device_train_batch_size` / `gradient_accumulation_steps` — start
   conservative (1/16 effective=16), bump after a dry-run if VRAM allows.
4. `data.max_seq_length` — keep at 4096 unless the model's context limit is lower.
5. `hub.repo_id` → your output Hub repo for the LoRA adapter.

Then dry-run before any full training:
```bash
PYTHONPATH=src python -m fine_tune.train --config src/fine_tune/configs/<new>.yaml --dry-run
```

If it produces a finite `[dry-run] OK. loss=…` line, you're ready to launch. If it
crashes, see `vlm-fine-tuning` skill troubleshooting catalog.
