#!/bin/bash
# Run all zero-shot benchmarks sequentially
# Usage: bash src/eval/run_all_benchmarks.sh
#
# Estimated time: ~3-4 hours on A40
# Estimated cost: ~$1.40

set -e

cd /workspace/dissertation
export HF_HOME=/workspace/hf_cache

echo "============================================"
echo "  Zero-Shot Benchmark Evaluation"
echo "  4 models × 3 benchmarks = 12 runs"
echo "============================================"

MODELS=("medgemma-4b" "gemma4-e4b" "qwen3.5-4b" "qwen3.5-9b")
BENCHMARKS=("fitzpatrick17k" "mm_skin_vqa" "confusion_triads")

START_TIME=$(date +%s)

for model in "${MODELS[@]}"; do
    for bench in "${BENCHMARKS[@]}"; do
        echo ""
        echo "============================================"
        echo "  $model | $bench"
        echo "  $(date)"
        echo "============================================"

        python3 src/eval/run_benchmark.py --model "$model" --benchmark "$bench"

        echo "  Done: $model | $bench"
    done

    echo ""
    echo ">>> Finished all benchmarks for $model"
    echo ""
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "============================================"
echo "  All benchmarks complete!"
echo "  Total time: ${ELAPSED} minutes"
echo "============================================"

# Score everything
echo ""
echo ">>> Scoring results..."
python3 src/eval/score_results.py

echo ""
echo ">>> Results saved to final/results/"
echo ">>> Download results to your Mac:"
echo '    scp -P <PORT> -r root@<IP>:/workspace/dissertation/final/results/ final/results/'
