#!/bin/bash
# Evaluate Qwen on ReasonOrderQA

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    source .env
fi

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATASET_PATH="${DATASET_PATH:-data/reasonorderqa/reasonorderqa.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qwen_reasonorderqa}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-10}"
BASE_URL="${BASE_URL:-https://api.siliconflow.cn/v1}"
API_KEY="${API_KEY:-${SILICONFLOW_API_KEY:-}}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUTPUT_DIR"

LIMIT_ARGS=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARGS="--limit $LIMIT"
fi

API_ARGS=""
if [ -n "$API_KEY" ]; then
    API_ARGS="--api_key $API_KEY"
fi

echo "=== Qwen ReasonOrderQA Evaluation ==="
echo "Model: $MODEL"
echo ""

python eval/run_qwen_eval_reasonorderqa.py \
    --model "$MODEL" \
    --dataset_path "$DATASET_PATH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --batch_size "$BATCH_SIZE" \
    --base_url "$BASE_URL" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "qwen_reasonorderqa_cot_first" \
    $API_ARGS $LIMIT_ARGS &
cot_pid=$!

python eval/run_qwen_eval_reasonorderqa.py \
    --model "$MODEL" \
    --dataset_path "$DATASET_PATH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --batch_size "$BATCH_SIZE" \
    --base_url "$BASE_URL" \
    --answer_first \
    --output_dir "$OUTPUT_DIR" \
    --run_name "qwen_reasonorderqa_answer_first" \
    $API_ARGS $LIMIT_ARGS &
ans_pid=$!

wait $cot_pid
wait $ans_pid

echo ""
echo "=== Results Summary ==="
for f in "$OUTPUT_DIR"/*_summary.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" _summary.json)
        accuracy=$(python3 -c "import json; print(f\"{json.load(open('$f'))['accuracy']:.2%}\")" 2>/dev/null || echo "N/A")
        echo "$name: $accuracy"
    fi
done
