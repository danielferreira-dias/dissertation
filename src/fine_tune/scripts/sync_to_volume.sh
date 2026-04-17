#!/usr/bin/env bash
# Upload final/train/ images + data/reasoning/reasoning.jsonl to the RunPod volume.
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

echo "Syncing final/train/ -> ${POD_ALIAS}:${REMOTE_ROOT}/images/final/train/"
rsync -avz --progress --partial \
  final/train/ \
  "${POD_ALIAS}:${REMOTE_ROOT}/images/final/train/"

echo "Syncing reasoning.jsonl -> ${POD_ALIAS}:${REMOTE_ROOT}/reasoning/"
ssh "${POD_ALIAS}" "mkdir -p ${REMOTE_ROOT}/reasoning"
rsync -avz --progress \
  data/reasoning/reasoning.jsonl \
  "${POD_ALIAS}:${REMOTE_ROOT}/reasoning/reasoning.jsonl"

echo "Sync complete."
echo "On pod: cd /workspace/dissertation && python -m fine_tune.prepare_data \\"
echo "          --source /workspace/data/reasoning/reasoning.jsonl \\"
echo "          --out-dir /workspace/data/reasoning \\"
echo "          --image-root /workspace/data/images"
