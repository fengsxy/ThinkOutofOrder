# 100-Sample Reproduction Plan

## Objective
Verify that the open-source code can reproduce the paper's key findings with a small sample (100 problems per dataset).

## Environment Setup

```bash
cd /home/ubuntu/efs/Retrieval_head/order-robustness-diffusion
source /home/ubuntu/efs/Retrieval_head/retrieval_head/.venv/bin/activate
```

## Experiments to Run

### 1. LLaDA on GSM8K (Diffusion vs AR, CoT vs Answer-First)

Expected: Diffusion mode should show similar accuracy for both prompt styles (order robust).
AR mode should show degradation for answer-first.

```bash
# Diffusion CoT-first
CUDA_VISIBLE_DEVICES=0 python eval/run_llada_eval_gsm8k.py \
    --model_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_path data/gsm8k \
    --limit 100 \
    --steps 256 --gen_length 256 --block_length 256 \
    --output_dir results/repro_100 \
    --run_name llada_gsm8k_diffusion_cot

# Diffusion Answer-first
CUDA_VISIBLE_DEVICES=0 python eval/run_llada_eval_gsm8k.py \
    --model_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_path data/gsm8k \
    --limit 100 \
    --steps 256 --gen_length 256 --block_length 256 \
    --answer_first \
    --output_dir results/repro_100 \
    --run_name llada_gsm8k_diffusion_ans

# AR CoT-first (block_length=1)
CUDA_VISIBLE_DEVICES=0 python eval/run_llada_eval_gsm8k.py \
    --model_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_path data/gsm8k \
    --limit 100 \
    --steps 256 --gen_length 256 --block_length 1 \
    --output_dir results/repro_100 \
    --run_name llada_gsm8k_ar_cot

# AR Answer-first
CUDA_VISIBLE_DEVICES=0 python eval/run_llada_eval_gsm8k.py \
    --model_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_path data/gsm8k \
    --limit 100 \
    --steps 256 --gen_length 256 --block_length 1 \
    --answer_first \
    --output_dir results/repro_100 \
    --run_name llada_gsm8k_ar_ans
```

### 2. LLaDA on ReasonOrderQA (Test per-difficulty breakdown)

Expected: D1 should be near-perfect, D2-D3 moderate, D4 very low.

```bash
# Diffusion CoT-first
CUDA_VISIBLE_DEVICES=0 python eval/run_llada_eval_reasonorderqa.py \
    --model_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_path data/reasonorderqa/reasonorderqa.jsonl \
    --limit 100 \
    --steps 256 --gen_length 256 --block_length 256 \
    --output_dir results/repro_100 \
    --run_name llada_roqa_diffusion_cot

# Diffusion Answer-first
CUDA_VISIBLE_DEVICES=0 python eval/run_llada_eval_reasonorderqa.py \
    --model_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_path data/reasonorderqa/reasonorderqa.jsonl \
    --limit 100 \
    --steps 256 --gen_length 256 --block_length 256 \
    --answer_first \
    --output_dir results/repro_100 \
    --run_name llada_roqa_diffusion_ans
```

### 3. Qwen Baseline on GSM8K (Should show large drop for answer-first)

```bash
# CoT-first
python eval/run_qwen_eval.py \
    --model Pro/Qwen/Qwen2.5-7B-Instruct \
    --dataset_path data/gsm8k \
    --limit 100 \
    --output_dir results/repro_100 \
    --run_name qwen_gsm8k_cot

# Answer-first
python eval/run_qwen_eval.py \
    --model Pro/Qwen/Qwen2.5-7B-Instruct \
    --dataset_path data/gsm8k \
    --limit 100 \
    --answer_first \
    --output_dir results/repro_100 \
    --run_name qwen_gsm8k_ans
```

## Success Criteria

| Experiment | Expected Pattern |
|------------|------------------|
| LLaDA Diffusion GSM8K | CoT ≈ Answer-first (< 10% relative drop) |
| LLaDA AR GSM8K | Answer-first << CoT (> 30% relative drop) |
| LLaDA ReasonOrderQA | D1 > 90%, D4 < 10% |
| Qwen GSM8K | Answer-first << CoT (> 50% relative drop) |

## Output Files

All results will be saved to `results/repro_100/`:
- `*_summary.json` - accuracy and config
- `*.jsonl` - per-problem predictions

## Verification Commands

```bash
# Check all summaries
for f in results/repro_100/*_summary.json; do
    echo "=== $(basename $f) ==="
    cat $f | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Accuracy: {d['accuracy']:.2%}\")"
done
```
