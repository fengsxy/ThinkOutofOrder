#!/bin/bash
# Evaluate Dream model on GSM8K, MATH500, and ReasonOrderQA

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    source .env
fi

MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dream}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
STEPS="${STEPS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
ALG="${ALG:-maskgit_plus}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUTPUT_DIR"

LIMIT_ARGS=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARGS="--limit $LIMIT"
fi

echo "=== Dream Model Evaluation ==="
echo "Model: $MODEL_PATH"
echo ""

# GSM8K
echo "Evaluating on GSM8K..."
python eval/run_dream_eval.py \
    --model_path "$MODEL_PATH" \
    --dataset_type gsm8k \
    --dataset_path data/gsm8k \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS" \
    --temperature "$TEMPERATURE" \
    --alg "$ALG" \
    --output_dir "$OUTPUT_DIR" \
    $LIMIT_ARGS &
gsm8k_diff_cot=$!

python eval/run_dream_eval.py \
    --model_path "$MODEL_PATH" \
    --dataset_type gsm8k \
    --dataset_path data/gsm8k \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS" \
    --temperature "$TEMPERATURE" \
    --alg "$ALG" \
    --answer_first \
    --output_dir "$OUTPUT_DIR" \
    $LIMIT_ARGS &
gsm8k_diff_ans=$!

wait $gsm8k_diff_cot
wait $gsm8k_diff_ans

# MATH500
echo "Evaluating on MATH500..."
python eval/run_dream_eval.py \
    --model_path "$MODEL_PATH" \
    --dataset_type math500 \
    --dataset_path data/math500/test.jsonl \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS" \
    --temperature "$TEMPERATURE" \
    --alg "$ALG" \
    --output_dir "$OUTPUT_DIR" \
    $LIMIT_ARGS &
math_diff_cot=$!

python eval/run_dream_eval.py \
    --model_path "$MODEL_PATH" \
    --dataset_type math500 \
    --dataset_path data/math500/test.jsonl \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS" \
    --temperature "$TEMPERATURE" \
    --alg "$ALG" \
    --answer_first \
    --output_dir "$OUTPUT_DIR" \
    $LIMIT_ARGS &
math_diff_ans=$!

wait $math_diff_cot
wait $math_diff_ans

# ReasonOrderQA
echo "Evaluating on ReasonOrderQA..."
python eval/run_dream_eval.py \
    --model_path "$MODEL_PATH" \
    --dataset_type reasonorderqa \
    --dataset_path data/reasonorderqa/reasonorderqa.jsonl \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS" \
    --temperature "$TEMPERATURE" \
    --alg "$ALG" \
    --output_dir "$OUTPUT_DIR" \
    $LIMIT_ARGS &
roqa_diff_cot=$!

python eval/run_dream_eval.py \
    --model_path "$MODEL_PATH" \
    --dataset_type reasonorderqa \
    --dataset_path data/reasonorderqa/reasonorderqa.jsonl \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS" \
    --temperature "$TEMPERATURE" \
    --alg "$ALG" \
    --answer_first \
    --output_dir "$OUTPUT_DIR" \
    $LIMIT_ARGS &
roqa_diff_ans=$!

wait $roqa_diff_cot
wait $roqa_diff_ans

echo ""
echo "=== Results Summary ==="
for f in "$OUTPUT_DIR"/dream_*_summary.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" _summary.json)
        accuracy=$(python3 -c "import json; print(f\"{json.load(open('$f'))['accuracy']:.2%}\")" 2>/dev/null || echo "N/A")
        echo "$name: $accuracy"
    fi
done
