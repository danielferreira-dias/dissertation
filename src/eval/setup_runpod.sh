#!/bin/bash
# RunPod A40 Setup Script
# Run this after SSH-ing into your RunPod instance
#
# Usage:
#   1. SSH into RunPod: ssh root@<your-runpod-ip> -p <port> -i ~/.ssh/id_ed25519
#   2. Run: bash setup_runpod.sh
#   3. Run benchmarks: bash run_all_benchmarks.sh

set -e

echo "============================================"
echo "  RunPod Setup — Dermatology VLM Evaluation"
echo "============================================"

# ── Step 1: System packages ─────────────────────────────────────────────────
echo ""
echo ">>> Step 1: System packages"
apt-get update -qq && apt-get install -y -qq git git-lfs unzip > /dev/null 2>&1
echo "Done."

# ── Step 2: Python dependencies ─────────────────────────────────────────────
echo ""
echo ">>> Step 2: Python dependencies"
pip install -q torch torchvision transformers accelerate \
    pillow pandas tqdm huggingface_hub sentencepiece \
    protobuf bitsandbytes scipy
echo "Done."

# ── Step 3: Clone project repo ──────────────────────────────────────────────
echo ""
echo ">>> Step 3: Project files"
# Option A: If you pushed to GitHub
# git clone https://github.com/<your-user>/dissertation.git /workspace/dissertation

# Option B: We'll create the structure and you rsync the files
mkdir -p /workspace/dissertation/src/eval
mkdir -p /workspace/dissertation/final/benchmarks
mkdir -p /workspace/dissertation/final/test
mkdir -p /workspace/dissertation/final/results

echo "Directory structure created."
echo ""
echo ">>> NOW UPLOAD YOUR FILES. Run this from your Mac (separate terminal):"
echo ""
echo '  RUNPOD="root@<YOUR_RUNPOD_IP> -p <PORT>"'
echo ""
echo "  # Upload eval scripts"
echo '  scp -P <PORT> -r src/eval/ root@<IP>:/workspace/dissertation/src/eval/'
echo ""
echo "  # Upload benchmarks"
echo '  scp -P <PORT> -r final/benchmarks/ root@<IP>:/workspace/dissertation/final/benchmarks/'
echo '  scp -P <PORT> -r final/test/ root@<IP>:/workspace/dissertation/final/test/'
echo ""
echo "  # Or use rsync (faster for large files):"
echo '  rsync -avz -e "ssh -p <PORT>" final/ root@<IP>:/workspace/dissertation/final/'
echo '  rsync -avz -e "ssh -p <PORT>" src/eval/ root@<IP>:/workspace/dissertation/src/eval/'
echo ""
echo "Press Enter after uploading files..."
read -r

# ── Step 4: Verify uploads ──────────────────────────────────────────────────
echo ""
echo ">>> Step 4: Verifying uploads"

verify() {
    local path=$1
    local label=$2
    if [ -d "$path" ]; then
        count=$(find "$path" -type f | wc -l)
        echo "  ✓ $label: $count files"
    else
        echo "  ✗ $label: MISSING"
    fi
}

verify "/workspace/dissertation/src/eval" "Eval scripts"
verify "/workspace/dissertation/final/test" "Confusion triads test set"
verify "/workspace/dissertation/final/benchmarks/fitzpatrick17k_1000" "Fitzpatrick17k benchmark"
verify "/workspace/dissertation/final/benchmarks/mm_skin/MM-SkinQA" "MM-Skin VQA images"
verify "/workspace/dissertation/final/benchmarks/mm_skin/vqa" "MM-Skin VQA metadata"

# ── Step 5: Pre-download models ─────────────────────────────────────────────
echo ""
echo ">>> Step 5: Pre-downloading models (this takes a while)"
python3 -c "
from huggingface_hub import snapshot_download
import os

models = [
    'google/medgemma-1.5-4b-it',
    'google/gemma-4-E4B-it',
    'Qwen/Qwen3.5-4B',
    'Qwen/Qwen3.5-9B',
]

for m in models:
    print(f'Downloading {m}...')
    try:
        snapshot_download(m, cache_dir='/workspace/hf_cache')
        print(f'  ✓ {m}')
    except Exception as e:
        print(f'  ✗ {m}: {e}')

os.environ['HF_HOME'] = '/workspace/hf_cache'
"

# ── Step 6: GPU check ───────────────────────────────────────────────────────
echo ""
echo ">>> Step 6: GPU check"
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "============================================"
echo "  Setup complete! Run benchmarks with:"
echo "  cd /workspace/dissertation"
echo "  bash src/eval/run_all_benchmarks.sh"
echo "============================================"
