#!/usr/bin/env bash
# Usage: bash src/fine_tune/scripts/run.sh src/fine_tune/configs/qwen3.5-9b.yaml [--format label_only] [--resume] [--dry-run]
set -euo pipefail

CONFIG="${1:?usage: run.sh <config> [flags...]}"
shift || true

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python -m fine_tune.train --config "${CONFIG}" "$@"
