#!/bin/bash
# Evaluate LLaDA on ReasonOrderQA

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    source .env
fi

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-data/reasonorderqa/reasonorderqa.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/llada_reasonorderqa}"
STEPS="${STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUTPUT_DIR"

LIMIT_ARGS=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARGS="--limit $LIMIT"
fi

# Prompt instructions for ReasonOrderQA
COT_INSTRUCTION='### YOUR TASK

You MUST output your answer in the following exact structure:

Retrieval:
(list all secret key numbers you found, in order)

Reasoning:
(explain how you combined the numbers to obtain the final result)

Answer:
(the final num ONLY, no extra text)'

ANSWER_INSTRUCTION='### YOUR TASK

You MUST output your answer in the following exact structure:

Answer:
(the final num ONLY, no extra text)
Reasoning:
(explain how you combined the numbers to obtain the final result)
Retrieval:
(list all secret key numbers you found, in order)'

echo "=== LLaDA ReasonOrderQA Evaluation ==="
echo "Model: $MODEL_PATH"
echo ""

# Diffusion mode
echo "Running Diffusion mode..."
python eval/run_llada_eval_reasonorderqa.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length "$GEN_LENGTH" \
    --temperature "$TEMPERATURE" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_reasonorderqa_diffusion_cot_first" \
    $LIMIT_ARGS &
diff_cot=$!

python eval/run_llada_eval_reasonorderqa.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length "$GEN_LENGTH" \
    --temperature "$TEMPERATURE" \
    --answer_first \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_reasonorderqa_diffusion_answer_first" \
    $LIMIT_ARGS &
diff_ans=$!

wait $diff_cot
wait $diff_ans

echo ""
echo "Running AR mode..."
python eval/run_llada_eval_reasonorderqa.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length 1 \
    --temperature "$TEMPERATURE" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_reasonorderqa_ar_cot_first" \
    $LIMIT_ARGS &
ar_cot=$!

python eval/run_llada_eval_reasonorderqa.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length 1 \
    --temperature "$TEMPERATURE" \
    --answer_first \
    --output_dir "$OUTPUT_DIR" \
    --run_name "llada_reasonorderqa_ar_answer_first" \
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
