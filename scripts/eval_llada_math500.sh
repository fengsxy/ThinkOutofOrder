#!/bin/bash
# Evaluate LLaDA on MATH500

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    source .env
fi

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-data/math500/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/llada_math500}"
STEPS="${STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUTPUT_DIR"

LIMIT_ARGS=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARGS="--limit $LIMIT"
fi

echo "=== LLaDA MATH500 Evaluation ==="
echo "Model: $MODEL_PATH"
echo ""

# Diffusion mode
python eval/run_llada_eval_gsm8k.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length "$GEN_LENGTH" \
    --temperature "$TEMPERATURE" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_math500_diffusion_cot_first" \
    $LIMIT_ARGS &
diff_cot=$!

python eval/run_llada_eval_gsm8k.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length "$GEN_LENGTH" \
    --temperature "$TEMPERATURE" \
    --answer_first \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_math500_diffusion_answer_first" \
    $LIMIT_ARGS &
diff_ans=$!

wait $diff_cot
wait $diff_ans

# AR mode
python eval/run_llada_eval_gsm8k.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length 1 \
    --temperature "$TEMPERATURE" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_math500_ar_cot_first" \
    $LIMIT_ARGS &
ar_cot=$!

python eval/run_llada_eval_gsm8k.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length 1 \
    --temperature "$TEMPERATURE" \
    --answer_first \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_math500_ar_answer_first" \
    $LIMIT_ARGS &
ar_ans=$!

wait $ar_cot
wait $ar_ans

echo ""
echo "=== Results Summary ==="
for f in "$OUTPUT_DIR"/*_summary.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" _summary.json)
        accuracy=$(python3 -c "import json; print(f\"{json.load(open('$f'))['accuracy']:.2%}\")" 2>/dev/null || echo "N/A")
        echo "$name: $accuracy"
    fi
done
