#!/usr/bin/env bash
# Install CUDA deps, HF login, verify GPU + data. Run once per fresh pod.
# HF_TOKEN must be set beforehand.
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
