#!/usr/bin/env python3
"""
Evaluate LLaDA discrete diffusion model on GSM8K.

Supports both diffusion mode (block_length > 1) and AR mode (block_length = 1),
with configurable prompt formats (CoT-first vs Answer-first).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.llada import LLaDAConfig, LLaDAModelLM

# Prompt Templates
DEFAULT_PROMPT_PREFIX = (
    "You are a careful math tutor who solves GSM8K problems. "
    "Provide the correct answer and explain your reasoning."
)
DEFAULT_PROMPT_SUFFIX = "Work through the question deliberately and ensure the final answer is correct."

COT_FIRST_INSTRUCTION = '''Explain the solution with a careful chain-of-thought before giving the final numeric result, and report the final answer inside \\boxed{}.
You MUST output your answer in the following exact structure:

Reasoning:
(explain how you combined the numbers to obtain the final result)

Answer:
\\boxed{number}'''

ANSWER_FIRST_INSTRUCTION = '''Begin by stating the final numeric answer inside \\boxed{} before giving the detailed chain-of-thought explanation.
You MUST start the response with the literal text "Answer:" on its own line (no text before it).
Immediately after the heading, output the numeric answer wrapped in \\boxed{}.
Then leave a blank line and provide the reasoning under a "Reasoning:" heading.

Answer:
\\boxed{number}

Reasoning:
(explain how you combined the numbers to obtain the final result)'''


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise for sampling."""
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute number of tokens to unmask at each step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    plan = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for row in range(mask_num.size(0)):
        extra = int(remainder[row].item())
        if extra > 0:
            plan[row, :extra] += 1
    return plan


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract numeric answer from model output."""
    if not text:
        return None
    # Try boxed format first
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        content = boxed_match.group(1).strip()
        num_match = re.search(r"-?\d[\d,]*\.?\d*", content.replace(",", ""))
        if num_match:
            return num_match.group(0)
    # Try "#### number" format (GSM8K style)
    hash_match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if hash_match:
        return hash_match.group(1).replace(",", "")
    # Fall back to last number in text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def parse_gold_answer(answer_text: str) -> str:
    """Parse gold answer from GSM8K format."""
    match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return answer_text.strip()


def build_prompt(question: str, prefix: str = None, suffix: str = None, instruction: str = None) -> str:
    """Build prompt from components."""
    parts = []
    if prefix:
        parts.append(prefix.strip())
    parts.append(f"Question: {question.strip()}")
    if suffix:
        parts.append(suffix.strip())
    if instruction:
        parts.append(instruction.strip())
    return "\n\n".join(parts)


@torch.no_grad()
def llada_decode(
    model: LLaDAModelLM,
    prompt_ids: torch.Tensor,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_token_id: int = 126336,
    zero_eos_confidence: bool = True,
    eos_token_id: int = 151643,
) -> torch.Tensor:
    """Run LLaDA diffusion decoding."""
    device = prompt_ids.device
    batch_size = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]

    # Initialize with mask tokens
    x = torch.full((batch_size, prompt_len + gen_length), mask_token_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids

    # Process in blocks
    num_blocks = (gen_length + block_length - 1) // block_length

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = min(prompt_len + (block_idx + 1) * block_length, prompt_len + gen_length)
        actual_block_len = block_end - block_start

        # Get mask indices for this block
        mask_index = (x[:, block_start:block_end] == mask_token_id)

        # Compute transfer schedule
        num_transfer = get_num_transfer_tokens(mask_index, steps)

        for step in range(steps):
            # Forward pass
            logits = model(x).logits
            block_logits = logits[:, block_start:block_end, :]

            # Apply temperature
            if temperature > 0:
                block_logits = add_gumbel_noise(block_logits, temperature)

            # Get confidence scores
            probs = F.softmax(block_logits.float(), dim=-1)
            confidence, pred_tokens = probs.max(dim=-1)

            # Zero out EOS confidence to prevent early stopping
            if zero_eos_confidence:
                eos_mask = (pred_tokens == eos_token_id)
                confidence = confidence.masked_fill(eos_mask, 0.0)

            # Only consider masked positions
            confidence = confidence * mask_index.float()

            # Select tokens to unmask based on remasking strategy
            n_unmask = num_transfer[:, step].item()
            if n_unmask > 0:
                if remasking == "low_confidence":
                    # Unmask highest confidence tokens
                    _, indices = confidence.topk(n_unmask, dim=-1)
                else:
                    # Random selection
                    masked_positions = mask_index.nonzero(as_tuple=True)[1]
                    perm = torch.randperm(len(masked_positions))[:n_unmask]
                    indices = masked_positions[perm].unsqueeze(0)

                # Update tokens
                for b in range(batch_size):
                    for idx in indices[b]:
                        x[b, block_start + idx] = pred_tokens[b, idx]
                        mask_index[b, idx] = False

    return x


def load_gsm8k(dataset_path: Path, split: str = "test", limit: int = None) -> List[Dict]:
    """Load GSM8K dataset."""
    jsonl_path = dataset_path / f"{split}.jsonl"
    if jsonl_path.exists():
        problems = []
        with open(jsonl_path) as f:
            for line in f:
                problems.append(json.loads(line))
        if limit:
            problems = problems[:limit]
        return problems
    # Fall back to HuggingFace
    ds = load_dataset("openai/gsm8k", "main", split=split)
    problems = [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]
    if limit:
        problems = problems[:limit]
    return problems


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaDA on GSM8K")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset_path", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--answer_first", action="store_true", help="Use answer-first prompt")
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine mode and prompt style
    mode = "ar" if args.block_length == 1 else "diffusion"
    prompt_style = "answer_first" if args.answer_first else "cot_first"
    instruction = ANSWER_FIRST_INSTRUCTION if args.answer_first else COT_FIRST_INSTRUCTION

    run_name = args.run_name or f"llada_gsm8k_{mode}_{prompt_style}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / f"{run_name}.jsonl"
    summary_json = args.output_dir / f"{run_name}_summary.json"

    print(f"Loading model from {args.model_path}...")
    config = LLaDAConfig.from_pretrained(args.model_path)
    model = LLaDAModelLM.from_pretrained(args.model_path, config=config, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading GSM8K from {args.dataset_path}...")
    problems = load_gsm8k(args.dataset_path, args.split, args.limit)
    print(f"Evaluating {len(problems)} problems ({mode} mode, {prompt_style})")

    correct = 0
    results = []

    with open(output_jsonl, "w") as writer:
        for idx, problem in enumerate(problems):
            question = problem["question"]
            gold_answer = parse_gold_answer(problem["answer"])

            prompt = build_prompt(question, DEFAULT_PROMPT_PREFIX, DEFAULT_PROMPT_SUFFIX, instruction)
            messages = [{"role": "user", "content": prompt}]
            chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_ids = tokenizer(chat_text, return_tensors="pt")["input_ids"].to(device)

            output_ids = llada_decode(
                model, prompt_ids, args.steps, args.gen_length, args.block_length,
                args.temperature, args.remasking
            )
            completion = tokenizer.decode(output_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
            pred_answer = extract_numeric_answer(completion)

            is_correct = pred_answer is not None and str(pred_answer) == str(gold_answer)
            if is_correct:
                correct += 1

            record = {"idx": idx, "question": question, "gold": gold_answer, "pred": pred_answer,
                      "correct": is_correct, "completion": completion}
            results.append(record)
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{idx+1}/{len(problems)}] pred={pred_answer} gold={gold_answer} correct={is_correct}")

    accuracy = correct / len(problems)
    summary = {"run_name": run_name, "accuracy": accuracy, "correct": correct, "total": len(problems),
               "mode": mode, "prompt_style": prompt_style, "steps": args.steps, "block_length": args.block_length}
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(problems)})")


if __name__ == "__main__":
    main()
