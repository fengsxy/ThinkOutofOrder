#!/bin/bash
# Evaluate LLaDA on GSM8K: diffusion vs AR, CoT-first vs Answer-first

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

# Load environment
if [ -f ".env" ]; then
    source .env
fi

# Configuration
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-data/gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-results/llada_gsm8k}"
STEPS="${STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUTPUT_DIR"

LIMIT_ARGS=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARGS="--limit $LIMIT"
fi

echo "=== LLaDA GSM8K Evaluation ==="
echo "Model: $MODEL_PATH"
echo "Steps: $STEPS, Gen Length: $GEN_LENGTH"
echo ""

# Diffusion mode (block_length = gen_length)
echo "Running Diffusion mode..."

# CoT-first
python eval/run_llada_eval_gsm8k.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length "$GEN_LENGTH" \
    --temperature "$TEMPERATURE" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_gsm8k_diffusion_cot_first" \
    $LIMIT_ARGS &
diffusion_cot_pid=$!

# Answer-first
python eval/run_llada_eval_gsm8k.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length "$GEN_LENGTH" \
    --temperature "$TEMPERATURE" \
    --answer_first \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_gsm8k_diffusion_answer_first" \
    $LIMIT_ARGS &
diffusion_ans_pid=$!

wait $diffusion_cot_pid
wait $diffusion_ans_pid

echo ""
echo "Running AR mode (block_length=1)..."

# AR mode (block_length = 1)
# CoT-first
python eval/run_llada_eval_gsm8k.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length 1 \
    --temperature "$TEMPERATURE" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_gsm8k_ar_cot_first" \
    $LIMIT_ARGS &
ar_cot_pid=$!

# Answer-first
python eval/run_llada_eval_gsm8k.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length 1 \
    --temperature "$TEMPERATURE" \
    --answer_first \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_gsm8k_ar_answer_first" \
    $LIMIT_ARGS &
ar_ans_pid=$!

wait $ar_cot_pid
wait $ar_ans_pid

echo ""
echo "=== Results Summary ==="
for f in "$OUTPUT_DIR"/*_summary.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" _summary.json)
        accuracy=$(python3 -c "import json; print(f\"{json.load(open('$f'))['accuracy']:.2%}\")" 2>/dev/null || echo "N/A")
        echo "$name: $accuracy"
    fi
done
