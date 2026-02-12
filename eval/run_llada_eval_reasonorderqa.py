#!/usr/bin/env python3
"""
Evaluate LLaDA discrete diffusion model on ReasonOrderQA.

ReasonOrderQA tests retrieval + reasoning with graded difficulty (D1-D4).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.llada import LLaDAConfig, LLaDAModelLM

# Prompt Templates for ReasonOrderQA
COT_FIRST_INSTRUCTION = '''### YOUR TASK

You MUST output your answer in the following exact structure:

Retrieval:
(list all secret key numbers you found, in order)

Reasoning:
(explain how you combined the numbers to obtain the final result)

Answer:
(the final num ONLY, no extra text)'''

ANSWER_FIRST_INSTRUCTION = '''### YOUR TASK

You MUST output your answer in the following exact structure:

Answer:
(the final num ONLY, no extra text)
Reasoning:
(explain how you combined the numbers to obtain the final result)
Retrieval:
(list all secret key numbers you found, in order)'''


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    plan = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for row in range(mask_num.size(0)):
        extra = int(remainder[row].item())
        if extra > 0:
            plan[row, :extra] += 1
    return plan


def extract_answer(text: str) -> Optional[str]:
    """Extract numeric answer from ReasonOrderQA output."""
    if not text:
        return None
    # Try Answer: section
    match = re.search(r"Answer:\s*(.+?)(?=\n(?:Reasoning|Retrieval)\b|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        chunk = match.group(1).strip()
        numbers = re.findall(r"-?\d+\.?\d*", chunk)
        if numbers:
            return numbers[-1]
    # Fall back to last number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return None


def extract_retrieved_numbers(text: str) -> List[int]:
    """Extract numbers from Retrieval section."""
    match = re.search(r"Retrieval:\s*(.+?)(?=\n(?:Reasoning|Answer)\b|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        chunk = match.group(1)
        numbers = re.findall(r"\d+", chunk)
        return [int(n) for n in numbers]
    return []


def compute_retrieval_f1(pred_keys: List[int], gold_keys: List[int]) -> float:
    """Compute F1 score for retrieved keys."""
    if not pred_keys or not gold_keys:
        return 0.0
    pred_set = set(pred_keys)
    gold_set = set(gold_keys)
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gold_set) if gold_set else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


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
    device = prompt_ids.device
    batch_size = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]

    x = torch.full((batch_size, prompt_len + gen_length), mask_token_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids

    num_blocks = (gen_length + block_length - 1) // block_length

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = min(prompt_len + (block_idx + 1) * block_length, prompt_len + gen_length)

        mask_index = (x[:, block_start:block_end] == mask_token_id)
        num_transfer = get_num_transfer_tokens(mask_index, steps)

        for step in range(steps):
            logits = model(x).logits
            block_logits = logits[:, block_start:block_end, :]

            if temperature > 0:
                block_logits = add_gumbel_noise(block_logits, temperature)

            probs = F.softmax(block_logits.float(), dim=-1)
            confidence, pred_tokens = probs.max(dim=-1)

            if zero_eos_confidence:
                eos_mask = (pred_tokens == eos_token_id)
                confidence = confidence.masked_fill(eos_mask, 0.0)

            confidence = confidence * mask_index.float()

            n_unmask = num_transfer[:, step].item()
            if n_unmask > 0:
                if remasking == "low_confidence":
                    _, indices = confidence.topk(n_unmask, dim=-1)
                else:
                    masked_positions = mask_index.nonzero(as_tuple=True)[1]
                    perm = torch.randperm(len(masked_positions))[:n_unmask]
                    indices = masked_positions[perm].unsqueeze(0)

                for b in range(batch_size):
                    for idx in indices[b]:
                        x[b, block_start + idx] = pred_tokens[b, idx]
                        mask_index[b, idx] = False

    return x


def load_reasonorderqa(dataset_path: Path, limit: int = None) -> List[Dict]:
    """Load ReasonOrderQA dataset."""
    problems = []
    with open(dataset_path) as f:
        for line in f:
            problems.append(json.loads(line))
    if limit:
        problems = problems[:limit]
    return problems


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaDA on ReasonOrderQA")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset_path", type=Path, default=Path("data/reasonorderqa/reasonorderqa.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence")
    parser.add_argument("--answer_first", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mode = "ar" if args.block_length == 1 else "diffusion"
    prompt_style = "answer_first" if args.answer_first else "cot_first"
    instruction = ANSWER_FIRST_INSTRUCTION if args.answer_first else COT_FIRST_INSTRUCTION

    run_name = args.run_name or f"llada_reasonorderqa_{mode}_{prompt_style}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / f"{run_name}.jsonl"
    summary_json = args.output_dir / f"{run_name}_summary.json"

    print(f"Loading model from {args.model_path}...")
    config = LLaDAConfig.from_pretrained(args.model_path)
    model = LLaDAModelLM.from_pretrained(args.model_path, config=config, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading ReasonOrderQA from {args.dataset_path}...")
    problems = load_reasonorderqa(args.dataset_path, args.limit)
    print(f"Evaluating {len(problems)} problems ({mode} mode, {prompt_style})")

    correct = 0
    total_f1 = 0.0
    results_by_difficulty = {1: [], 2: [], 3: [], 4: []}

    with open(output_jsonl, "w") as writer:
        for idx, problem in enumerate(problems):
            question = problem["question"]
            gold_answer = str(problem["answer"])
            metadata = problem.get("metadata", {})
            gold_keys = metadata.get("gold_keys", [])
            difficulty = metadata.get("difficulty", 0)

            prompt = f"{question}\n\n{instruction}"
            messages = [{"role": "user", "content": prompt}]
            chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_ids = tokenizer(chat_text, return_tensors="pt")["input_ids"].to(device)

            output_ids = llada_decode(
                model, prompt_ids, args.steps, args.gen_length, args.block_length,
                args.temperature, args.remasking
            )
            completion = tokenizer.decode(output_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)

            pred_answer = extract_answer(completion)
            pred_keys = extract_retrieved_numbers(completion)
            retrieval_f1 = compute_retrieval_f1(pred_keys, gold_keys)

            is_correct = pred_answer is not None and str(pred_answer) == gold_answer
            if is_correct:
                correct += 1
            total_f1 += retrieval_f1

            record = {
                "idx": idx,
                "difficulty": difficulty,
                "gold": gold_answer,
                "pred": pred_answer,
                "correct": is_correct,
                "gold_keys": gold_keys,
                "pred_keys": pred_keys,
                "retrieval_f1": retrieval_f1,
                "completion": completion,
            }
            results_by_difficulty[difficulty].append(record)
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{idx+1}/{len(problems)}] D{difficulty} pred={pred_answer} gold={gold_answer} correct={is_correct} f1={retrieval_f1:.2f}")

    accuracy = correct / len(problems)
    mean_f1 = total_f1 / len(problems)

    # Compute per-difficulty stats
    difficulty_stats = {}
    for d in [1, 2, 3, 4]:
        if results_by_difficulty[d]:
            d_correct = sum(1 for r in results_by_difficulty[d] if r["correct"])
            d_total = len(results_by_difficulty[d])
            difficulty_stats[f"D{d}_accuracy"] = d_correct / d_total
            difficulty_stats[f"D{d}_count"] = d_total

    summary = {
        "run_name": run_name,
        "accuracy": accuracy,
        "mean_retrieval_f1": mean_f1,
        "correct": correct,
        "total": len(problems),
        "mode": mode,
        "prompt_style": prompt_style,
        **difficulty_stats,
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{len(problems)})")
    print(f"Mean Retrieval F1: {mean_f1:.3f}")
    for d in [1, 2, 3, 4]:
        if f"D{d}_accuracy" in difficulty_stats:
            print(f"  D{d}: {difficulty_stats[f'D{d}_accuracy']:.2%} ({difficulty_stats[f'D{d}_count']} problems)")


if __name__ == "__main__":
    main()
