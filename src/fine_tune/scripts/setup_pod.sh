#!/usr/bin/env bash
# Build a reproducible VLM-fine-tune environment on a fresh RunPod pod
# (Unsloth-based). Idempotent: safe to re-run after a pod restart.
#
# Designed for driver R565 ("CUDA 12.7" reported by nvidia-smi), which
# supports CUDA toolkit ≤ 12.6. We use Unsloth's `cu124-torch260` extra to
# pull a known-good combo with all kernels (FA2, fla, triton, bnb) pre-built
# against the pinned torch.
#
# Required env vars before running:
#   HF_TOKEN            — write-scope HF token (huggingface.co/settings/tokens)
#
# Usage:
#   export HF_TOKEN=hf_…
#   bash src/fine_tune/scripts/setup_pod.sh
#
# Sets these afterwards (also in /etc/profile.d so they survive new shells):
#   HF_HOME=/workspace/hf_cache
#   HF_HUB_DISABLE_XET=1                       (Xet downloads are flaky on rented pods)
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (avoids fragmentation OOMs)
#   UNSLOTH_RETURN_LOGITS=1                    (Unsloth's recommended logit handling)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
REQS_GPU="${REPO_ROOT}/src/fine_tune/requirements-gpu.txt"

# ── 0. sanity ────────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set. export HF_TOKEN=hf_… and re-run." >&2
  exit 1
fi
[ -f "$REQS_GPU" ] || { echo "ERROR: missing $REQS_GPU" >&2; exit 1; }

# ── 1. persistent dirs + env vars ────────────────────────────────────────
mkdir -p /workspace/hf_cache /workspace/runs /workspace/pip_cache

cat > /etc/profile.d/dissertation_env.sh << 'EOF'
export HF_HOME=/workspace/hf_cache
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNSLOTH_RETURN_LOGITS=1
EOF

export HF_HOME=/workspace/hf_cache
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNSLOTH_RETURN_LOGITS=1

# ── 2. apt: tmux + rsync (containers reset on restart) ──────────────────
apt-get update -qq
apt-get install -y -qq tmux rsync

# ── 3. install unsloth + pure-Python deps ────────────────────────────────
echo
echo "─── pip install unsloth + deps ───"
# Use --upgrade so an old torch from a previous setup_pod.sh run is replaced
# with the version Unsloth's extra wants.
pip install --cache-dir /workspace/pip_cache --upgrade -r "$REQS_GPU"

# ── 4. HF login ──────────────────────────────────────────────────────────
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# ── 5. validate ──────────────────────────────────────────────────────────
echo
echo "─── env validation ───"
python "${REPO_ROOT}/src/fine_tune/scripts/verify_env.py"

echo
echo "Setup complete. Source the env vars in new shells with:"
echo "  source /etc/profile.d/dissertation_env.sh"
