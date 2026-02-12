# Thinking Out of Order

This repository contains the code and data for reproducing the experiments in:

**"Thinking Out of Order: When Output Order Stops Reflecting Reasoning Order in Diffusion Language Models"**

Longxuan Yu, Yu Fu, Shaorong Zhang, Hui Liu, Mukund Varma T, Greg Ver Steeg, Yue Dong

[[arXiv]](https://arxiv.org/abs/2601.22035)

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
│   ├── run_llada_sampling_methods.py  # Sampling strategy comparison
│   ├── run_dream_eval.py
│   └── run_qwen_eval.py
├── analysis/
│   └── plot_confidence_heatmap.py     # D1/D4 confidence heatmap
├── scripts/
│   ├── eval_llada_gsm8k.sh
│   ├── eval_llada_math500.sh
│   ├── eval_llada_reasonorderqa.sh
│   ├── eval_sampling_methods.sh       # All sampling methods
│   ├── eval_dream.sh
│   └── eval_qwen_gsm8k.sh
├── figures/                           # Generated plots
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

### Sampling Methods Comparison

Compare different remasking strategies (low_confidence, high_confidence, random, topk_margin, entropy, left_to_right):

```bash
# Run all sampling methods
bash scripts/eval_sampling_methods.sh

# Or run individual method
python eval/run_llada_sampling_methods.py \
    --remasking low_confidence \
    --steps 256 \
    --gen_length 64 \
    --block_length 64 \
    --output_dir results/sampling_method
```

### Confidence Heatmap (Figure 3)

Generate confidence trace data and plot heatmap:

```bash
# Step 1: Run with --save_trace to capture confidence matrices
python eval/run_llada_sampling_methods.py \
    --remasking low_confidence \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --save_trace \
    --output_dir results/confidence_trace \
    --run_name low_confidence_cot

# Step 2: Plot D1/D4 combined heatmap
python analysis/plot_confidence_heatmap.py \
    --input results/confidence_trace/low_confidence_cot_trace/trace.jsonl \
    --output figures/heatmap_d1_d4.png
```

Key parameters for heatmap:
- `--save_trace`: Save per-step confidence matrices
- `--steps 256 --gen_length 256 --block_length 256`: Generate 256x256 matrix for full visualization

## Models

| Model | Type | HuggingFace |
|-------|------|-------------|
| LLaDA-8B-Instruct | Diffusion | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) |
| Dream-v0-Instruct-7B | Diffusion | [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) |
| Qwen2.5-7B-Instruct | AR | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |

## Citation

```bibtex
@article{yu2025thinking,
  title={Thinking Out of Order: When Output Order Stops Reflecting Reasoning Order in Diffusion Language Models},
  author={Yu, Longxuan and Fu, Yu and Zhang, Shaorong and Liu, Hui and Varma T, Mukund and Ver Steeg, Greg and Dong, Yue},
  journal={arXiv preprint arXiv:2601.22035},
  year={2025}
}
```

## License

MIT License
# ThinkOutofOrder
