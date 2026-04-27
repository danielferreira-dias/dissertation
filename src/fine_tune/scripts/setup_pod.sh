#!/usr/bin/env bash
# Build a reproducible VLM-fine-tune environment on a fresh RunPod pod.
#
# Idempotent: safe to re-run after a pod restart. Designed for driver R565 ("CUDA 12.7"
# reported by nvidia-smi), which supports CUDA toolkit ≤ 12.6 — i.e. cu124 wheels are
# in-range, cu128/cu130 are not.
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

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
REQS_GPU="${REPO_ROOT}/src/fine_tune/requirements-gpu.txt"
REQS_KERNELS="${REPO_ROOT}/src/fine_tune/requirements-kernels.txt"

# ── 0. sanity ────────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set. export HF_TOKEN=hf_… and re-run." >&2
  exit 1
fi

[ -f "$REQS_GPU" ]     || { echo "ERROR: missing $REQS_GPU"     >&2; exit 1; }
[ -f "$REQS_KERNELS" ] || { echo "ERROR: missing $REQS_KERNELS" >&2; exit 1; }

# ── 1. persistent dirs + env vars ────────────────────────────────────────
mkdir -p /workspace/hf_cache /workspace/runs /workspace/pip_cache

cat > /etc/profile.d/dissertation_env.sh << 'EOF'
export HF_HOME=/workspace/hf_cache
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

# Source for the current shell too
export HF_HOME=/workspace/hf_cache
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 2. apt: tmux + rsync (containers reset on restart) ──────────────────
apt-get update -qq
apt-get install -y -qq tmux rsync

# ── 3. pinned torch + transformers stack ────────────────────────────────
echo
echo "─── pip install pinned GPU stack ───"
pip install --cache-dir /workspace/pip_cache --upgrade -r "$REQS_GPU"

# ── 4. CUDA kernels (built against the just-installed torch) ────────────
echo
echo "─── pip install kernels (no-build-isolation) ───"
pip install --cache-dir /workspace/pip_cache --no-build-isolation -r "$REQS_KERNELS"

# ── 5. HF login ──────────────────────────────────────────────────────────
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# ── 6. validate ──────────────────────────────────────────────────────────
echo
echo "─── env validation ───"
python "${REPO_ROOT}/src/fine_tune/scripts/verify_env.py"

echo
echo "Setup complete. Source the env vars in new shells with:"
echo "  source /etc/profile.d/dissertation_env.sh"
