#!/bin/bash
# Phase 1: Ablation study on Qwen 3.5 4B
# 4 runs per backend: {LoRA, QLoRA} × {label_only, full_reasoning}
#
# Run on GPU machine (RunPod L40S 48GB):
#   BACKEND=unsloth bash src/fine_tune/run_phase1.sh   # recommended (faster)
#   BACKEND=peft    bash src/fine_tune/run_phase1.sh   # standard fallback
#
# Prerequisites:
#   # For PEFT backend:
#   pip install torch transformers peft bitsandbytes datasets accelerate pillow
#   # For Unsloth backend:
#   pip install unsloth trl
#
#   python src/fine_tune/prepare_data.py  # generate training data first

set -e

BACKEND="${BACKEND:-unsloth}"
MODEL="qwen3.5-4b"

if [ "$BACKEND" = "unsloth" ]; then
    SCRIPT="src/fine_tune/train_unsloth.py"
    echo "=== Phase 1: $MODEL Ablation (Unsloth backend) ==="
elif [ "$BACKEND" = "peft" ]; then
    SCRIPT="src/fine_tune/train.py"
    echo "=== Phase 1: $MODEL Ablation (HF PEFT backend) ==="
else
    echo "Error: BACKEND must be 'unsloth' or 'peft' (got: $BACKEND)"
    exit 1
fi
echo ""

# Common training flags
COMMON_ARGS="--model $MODEL --epochs 3 --lr 2e-4"

# Run 1: LoRA + label_only
echo ">>> Run 1/4: LoRA + label_only"
python $SCRIPT $COMMON_ARGS \
    --data-format label_only --method lora \
    --batch-size 2 --grad-accum 8

# Run 2: LoRA + full_reasoning
echo ">>> Run 2/4: LoRA + full_reasoning"
python $SCRIPT $COMMON_ARGS \
    --data-format full_reasoning --method lora \
    --batch-size 2 --grad-accum 8

# Run 3: QLoRA + label_only
echo ">>> Run 3/4: QLoRA + label_only"
python $SCRIPT $COMMON_ARGS \
    --data-format label_only --method qlora \
    --batch-size 4 --grad-accum 4

# Run 4: QLoRA + full_reasoning
echo ">>> Run 4/4: QLoRA + full_reasoning"
python $SCRIPT $COMMON_ARGS \
    --data-format full_reasoning --method qlora \
    --batch-size 4 --grad-accum 4

echo ""
echo "=== Phase 1 complete ($BACKEND backend) ==="
echo "Results in models/${BACKEND:+unsloth-}$MODEL-{label_only,full_reasoning}-{lora,qlora}/"
