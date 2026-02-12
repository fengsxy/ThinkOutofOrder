# Order Robustness in Discrete Diffusion Language Models

This repository contains the code and data for reproducing the experiments in:

**"Order Robustness in Discrete Diffusion Language Models: A Confidence-Based Analysis"**

## Key Findings

We investigate whether discrete diffusion language models (dLLMs) exhibit **order robustness** — the ability to produce correct answers regardless of whether the prompt requests reasoning-first (CoT) or answer-first output formats.

| Model | Type | GSM8K Drop | MATH500 Drop | Order Robust? |
|-------|------|------------|--------------|---------------|
| Qwen2.5-7B | Autoregressive | ~67% | ~50% | No |
| Dream-7B | Diffusion (distilled) | ~46% | ~30% | Partial |
| LLaDA-8B | Diffusion (scratch) | ~4% | ~5% | Yes |

Key insight: Confidence-based remasking in diffusion models naturally defers high-uncertainty answer tokens, allowing reasoning to emerge before answers regardless of output position.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/order-robustness-diffusion.git
cd order-robustness-diffusion
pip install -r requirements.txt

# Copy .env.example to .env and add your API key (for Qwen evaluation)
cp .env.example .env

# Download datasets
bash scripts/download_data.sh

# Generate ReasonOrderQA (or use provided data)
python data/generate_reasonorderqa.py --count 1000 --seed 42

# Run experiments
bash scripts/eval_llada_gsm8k.sh      # LLaDA on GSM8K
bash scripts/eval_dream.sh            # Dream on all datasets
bash scripts/eval_qwen_gsm8k.sh       # Qwen baseline
```

## Project Structure

```
order-robustness-diffusion/
├── data/
│   ├── gsm8k/                    # GSM8K test set
│   ├── math500/                  # MATH500 test set
│   ├── reasonorderqa/            # ReasonOrderQA benchmark
│   └── generate_reasonorderqa.py # Dataset generator
├── models/
│   └── llada/                    # LLaDA model implementation
├── eval/
│   ├── run_llada_eval_gsm8k.py
│   ├── run_llada_eval_reasonorderqa.py
│   ├── run_dream_eval.py
│   └── run_qwen_eval.py
├── scripts/
│   ├── eval_llada_gsm8k.sh
│   ├── eval_llada_math500.sh
│   ├── eval_llada_reasonorderqa.sh
│   ├── eval_dream.sh
│   └── eval_qwen_gsm8k.sh
└── results/
```

## Experimental Settings

Following the paper, we use these default hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Gen Length ($L_{gen}$) | 256 | Maximum generation length |
| Block Size ($L_{block}$) | 256 | Diffusion block size (1 = AR mode) |
| Diff Steps ($T$) | 256 | Number of diffusion steps |
| Stepwise Rate | 1 | $L_{gen}/T$ = 1 token/step |
| Temperature | 0.0 | Deterministic decoding |
| Remasking | low_confidence | Confidence-based remasking |

## ReasonOrderQA Benchmark

ReasonOrderQA is a controlled benchmark with graded arithmetic complexity:

| Level | Expression | Variables | Complexity |
|-------|------------|-----------|------------|
| D1 | X + Y + Z | 3 | Simple addition |
| D2 | X + Y - Z | 3 | Mixed operations |
| D3 | (X + Y) × Z | 3 | Parenthesized |
| D4 | (X - Y × Z) × W | 4 | Multi-step |

Distribution: D1:D2:D3:D4 = 0.25:0.40:0.25:0.10

Generate the dataset:
```bash
python data/generate_reasonorderqa.py \
    --count 1000 \
    --target_length 1000 \
    --seed 42 \
    --output data/reasonorderqa/reasonorderqa.jsonl
```

## Reproducing Paper Results

### Table 1: Main Results

```bash
# LLaDA (diffusion vs AR)
bash scripts/eval_llada_gsm8k.sh
bash scripts/eval_llada_math500.sh

# Dream
bash scripts/eval_dream.sh

# Qwen (AR baseline)
bash scripts/eval_qwen_gsm8k.sh
```

### Table 2: ReasonOrderQA by Difficulty

```bash
bash scripts/eval_llada_reasonorderqa.sh
```

### Confidence Analysis

To capture per-step confidence traces:
```bash
python eval/run_llada_eval_gsm8k.py \
    --model_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_path data/gsm8k \
    --full_trace_jsonl results/gsm8k_trace.jsonl
```

## Models

| Model | Type | HuggingFace |
|-------|------|-------------|
| LLaDA-8B-Instruct | Diffusion | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) |
| Dream-v0-Instruct-7B | Diffusion | [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) |
| Qwen2.5-7B-Instruct | AR | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |

## Citation

```bibtex
@inproceedings{author2026order,
  title={Order Robustness in Discrete Diffusion Language Models: A Confidence-Based Analysis},
  author={Author Names},
  booktitle={Proceedings of ICML 2026},
  year={2026}
}
```

## License

MIT License
# ThinkOutofOrder
