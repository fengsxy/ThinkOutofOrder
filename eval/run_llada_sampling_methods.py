#!/usr/bin/env python3
"""
Experiment script to compare different sampling algorithms on ReasonOrderQA.

Sampling algorithms:
1. high_confidence (baseline, same as Dream's maskgit_plus): Uses top1 probability as confidence, unmask highest first
2. random: Random confidence scores
3. topk_margin: Uses (top1 - top2) probability margin as confidence
4. entropy: Uses negative entropy as confidence

Usage:
    python eval/run_llada_sampling_methods.py --remasking high_confidence --output results/sampling_method/high_confidence.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.llada import LLaDAConfig, LLaDAModelLM


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise for sampling."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute the number of tokens to transfer at each step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def compute_confidence(
    logits: torch.Tensor,
    predictions: torch.Tensor,
    remasking: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute confidence scores based on the remasking strategy.

    Args:
        logits: Model logits [batch, seq_len, vocab]
        predictions: Predicted token ids [batch, seq_len]
        remasking: Strategy name
        device: Torch device

    Returns:
        Confidence scores [batch, seq_len]
    """
    batch_size, seq_len = predictions.shape

    if remasking in ("high_confidence", "low_confidence"):
        # Standard (same as Dream's maskgit_plus): confidence = P(predicted_token)
        probs = F.softmax(logits, dim=-1)
        gather_index = predictions.unsqueeze(-1)
        confidence = torch.squeeze(torch.gather(probs, dim=-1, index=gather_index), -1)

    elif remasking == "random":
        # Random confidence
        confidence = torch.rand((batch_size, seq_len), device=device)

    elif remasking == "topk_margin":
        # Confidence = P(top1) - P(top2)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, :, 0]
        top2_probs = sorted_probs[:, :, 1]
        confidence = top1_probs - top2_probs

    elif remasking == "entropy":
        # Confidence = negative entropy (higher = more certain)
        probs = F.softmax(logits, dim=-1)
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    elif remasking == "left_to_right":
        # Left-to-right: confidence = -position (leftmost first)
        confidence = -torch.arange(seq_len, device=device).float().unsqueeze(0).expand(batch_size, -1)

    else:
        raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

    return confidence


# Prompt templates
COT_INSTRUCTION = """### YOUR TASK

You MUST output your answer in the following exact structure:

Retrieval:
(list all secret key numbers you found, in order)

Reasoning:
(explain how you combined the numbers to obtain the final result)

Answer:
(the final num ONLY, no extra text)"""

ANSWER_FIRST_INSTRUCTION = """### YOUR TASK

You MUST output your answer in the following exact structure:

Answer:
(the final num ONLY, no extra text)
Reasoning:
(explain how you combined the numbers to obtain the final result)
Retrieval:
(list all secret key numbers you found, in order)"""


def load_problems(path: Path) -> List[Dict[str, Any]]:
    """Load problems from JSONL file."""
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def build_prompt_ids(
    problem: Dict[str, Any],
    tokenizer: AutoTokenizer,
    answer_first: bool = False,
) -> torch.Tensor:
    """Build prompt tensor from problem."""
    instruction = ANSWER_FIRST_INSTRUCTION if answer_first else COT_INSTRUCTION
    question = problem.get("question", "")
    prompt = f"{question}\n\n{instruction}"
    chat_messages = [{"role": "user", "content": prompt}]

    chat_text = tokenizer.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer(chat_text, return_tensors="pt")
    return encoded["input_ids"]


ANSWER_SECTION_PATTERN = re.compile(
    r"Answer:\s*(.+?)(?=\n(?:Reasoning|Retrieval)\b|$)", re.IGNORECASE | re.DOTALL
)


def extract_answer(text: str) -> str | None:
    """Extract numeric answer from generated text."""
    if not text:
        return None
    match = ANSWER_SECTION_PATTERN.search(text)
    if match:
        chunk = match.group(1).strip()
        numbers = re.findall(r"-?\d+\.?\d*", chunk)
        if numbers:
            return numbers[-1]
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return None


def extract_retrieved_numbers(text: str) -> List[int]:
    """Extract retrieved key numbers from text."""
    match = re.search(r"Retrieval:\s*(.+?)(?=\n(?:Reasoning|Answer)\b|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        chunk = match.group(1)
        numbers = re.findall(r"\d+", chunk)
        return [int(n) for n in numbers]
    return []


def compute_retrieval_f1(pred_keys: List[int], gold_keys: List[int]) -> float:
    """Compute F1 score for retrieval."""
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
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str,
    mask_id: int,
    save_trace: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """LLaDA decoding with configurable sampling strategy."""
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    batch_size, prompt_len = prompt_ids.shape
    total_len = prompt_len + gen_length

    x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = steps // num_blocks

    step_traces = []

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step_idx in range(steps_per_block):
            mask_index = x == mask_id
            if not mask_index.any():
                break

            logits = model(input_ids=x, use_cache=False).logits
            logits_with_noise = add_gumbel_noise(logits, temperature)
            predictions = torch.argmax(logits_with_noise, dim=-1)

            # Compute confidence based on remasking strategy
            confidence = compute_confidence(logits, predictions, remasking, device)

            # Save trace
            if save_trace:
                global_step = block_idx * steps_per_block + step_idx
                gen_conf = confidence[0, prompt_len:].detach().cpu().tolist()
                step_traces.append({
                    "step": global_step,
                    "confidence_vector": gen_conf,
                    "masks_remaining": int(mask_index.sum().item()),
                })

            # Don't consider positions beyond current block
            confidence[:, block_end:] = float("-inf")

            # Only consider masked positions for transfer
            x0 = torch.where(mask_index, predictions, x)
            confidence = torch.where(mask_index, confidence, torch.full_like(confidence, float("-inf")))

            # Select top-k tokens to unmask
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for row in range(batch_size):
                quota = int(num_transfer_tokens[row, step_idx].item())
                if quota <= 0:
                    continue
                quota = min(quota, confidence.shape[1])
                _, idx = torch.topk(confidence[row], k=quota)
                transfer_index[row, idx] = True

            x[transfer_index] = x0[transfer_index]

    # Decode output
    output_ids = x[0, prompt_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return output_text, step_traces


def run_experiment(
    model: LLaDAModelLM,
    tokenizer: AutoTokenizer,
    problems: List[Dict[str, Any]],
    remasking: str,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    mask_id: int,
    output_path: Path,
    answer_first: bool = False,
    save_trace: bool = False,
    limit: int = None,
) -> Dict[str, float]:
    """Run experiment on all problems and return metrics."""
    if limit:
        problems = problems[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup trace directory if needed
    trace_file = None
    if save_trace:
        trace_dir = output_path.parent / (output_path.stem + "_trace")
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = open(trace_dir / "trace.jsonl", "w", encoding="utf-8")

    correct = 0
    total = 0
    total_f1 = 0.0
    prompt_mode = "answer_first" if answer_first else "cot_first"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, problem in enumerate(problems):
                prompt_ids = build_prompt_ids(problem, tokenizer, answer_first=answer_first)
                gold_answer = str(problem.get("answer", ""))
                metadata = problem.get("metadata", {})
                gold_keys = metadata.get("gold_keys", [])
                difficulty = metadata.get("difficulty", 0)

                completion, step_traces = llada_decode(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    remasking=remasking,
                    mask_id=mask_id,
                    save_trace=save_trace,
                )

                pred_answer = extract_answer(completion)
                pred_keys = extract_retrieved_numbers(completion)
                retrieval_f1 = compute_retrieval_f1(pred_keys, gold_keys)

                is_correct = pred_answer is not None and str(pred_answer) == gold_answer
                if is_correct:
                    correct += 1
                total += 1
                total_f1 += retrieval_f1

                record = {
                    "idx": idx,
                    "difficulty": difficulty,
                    "gold": gold_answer,
                    "pred": pred_answer,
                    "correct": is_correct,
                    "retrieval_f1": retrieval_f1,
                    "completion": completion,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Save trace with confidence_matrix for heatmap plotting
                if save_trace and trace_file:
                    # Convert step_traces to confidence_matrix [steps x gen_length]
                    if step_traces:
                        confidence_matrix = [st["confidence_vector"] for st in step_traces]
                    else:
                        confidence_matrix = []
                    trace_record = {
                        "index": idx,
                        "idx": idx,
                        "difficulty": difficulty,
                        "completion": completion,
                        "confidence_matrix": confidence_matrix,
                    }
                    trace_file.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

                print(f"[{idx+1}/{len(problems)}] D{difficulty} pred={pred_answer} gold={gold_answer} correct={is_correct}")

    finally:
        if trace_file:
            trace_file.close()

    accuracy = correct / total if total > 0 else 0
    mean_f1 = total_f1 / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "mean_retrieval_f1": mean_f1,
        "correct": correct,
        "total": total,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare sampling methods on ReasonOrderQA")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset_path", type=Path, default=Path("data/reasonorderqa/reasonorderqa.jsonl"))
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "high_confidence", "random", "topk_margin", "entropy", "left_to_right"])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--answer_first", action="store_true")
    parser.add_argument("--save_trace", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("results/sampling_method"))
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    prompt_style = "answer_first" if args.answer_first else "cot_first"
    run_name = args.run_name or f"{args.remasking}_{prompt_style}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / f"{run_name}.jsonl"
    summary_json = args.output_dir / f"{run_name}_summary.json"

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = 126336  # LLaDA default

    print(f"Loading ReasonOrderQA from {args.dataset_path}...")
    problems = load_problems(args.dataset_path)
    print(f"Evaluating {len(problems)} problems with {args.remasking} remasking ({prompt_style})")

    metrics = run_experiment(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        remasking=args.remasking,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        mask_id=mask_id,
        output_path=output_jsonl,
        answer_first=args.answer_first,
        save_trace=args.save_trace,
        limit=args.limit,
    )

    summary = {
        "run_name": run_name,
        "remasking": args.remasking,
        "prompt_style": prompt_style,
        **metrics,
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Mean Retrieval F1: {metrics['mean_retrieval_f1']:.3f}")


if __name__ == "__main__":
    main()
