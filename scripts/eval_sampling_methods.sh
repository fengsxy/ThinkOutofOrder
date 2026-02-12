#!/bin/bash
# Run all sampling method experiments on ReasonOrderQA
# Usage: bash scripts/eval_sampling_methods.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    source .env
fi

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-data/reasonorderqa/reasonorderqa.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/sampling_method}"
STEPS="${STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-64}"
BLOCK_LENGTH="${BLOCK_LENGTH:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUTPUT_DIR"

LIMIT_ARGS=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARGS="--limit $LIMIT"
fi

METHODS=("low_confidence" "high_confidence" "random" "topk_margin" "entropy" "left_to_right")

echo "=== Sampling Methods Experiment ==="
echo "Model: $MODEL_PATH"
echo "Methods: ${METHODS[*]}"
echo ""

# Run CoT-first experiments
echo "Running CoT-first experiments..."
for method in "${METHODS[@]}"; do
    echo "  Running $method (cot_first)..."
    python eval/run_llada_sampling_methods.py \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --remasking "$method" \
        --steps "$STEPS" \
        --gen_length "$GEN_LENGTH" \
        --block_length "$BLOCK_LENGTH" \
        --temperature "$TEMPERATURE" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "${method}_cot_first" \
        $LIMIT_ARGS
done

# Run Answer-first experiments
echo ""
echo "Running Answer-first experiments..."
for method in "${METHODS[@]}"; do
    echo "  Running $method (answer_first)..."
    python eval/run_llada_sampling_methods.py \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --remasking "$method" \
        --steps "$STEPS" \
        --gen_length "$GEN_LENGTH" \
        --block_length "$BLOCK_LENGTH" \
        --temperature "$TEMPERATURE" \
        --answer_first \
        --output_dir "$OUTPUT_DIR" \
        --run_name "${method}_answer_first" \
        $LIMIT_ARGS
done

echo ""
echo "=== Results Summary ==="
for f in "$OUTPUT_DIR"/*_summary.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" _summary.json)
        accuracy=$(python3 -c "import json; print(f\"{json.load(open('$f'))['accuracy']:.2%}\")" 2>/dev/null || echo "N/A")
        echo "$name: $accuracy"
    fi
done
