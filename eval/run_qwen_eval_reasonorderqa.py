#!/usr/bin/env python3
"""
Evaluate Qwen on ReasonOrderQA using batch API calls.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from gpt_batch.batcher import GPTBatcher


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

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"Answer:\s*(.+?)(?=\n(?:Reasoning|Retrieval)\b|$)", text, re.IGNORECASE | re.DOTALL)
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
    match = re.search(r"Retrieval:\s*(.+?)(?=\n(?:Reasoning|Answer)\b|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        chunk = match.group(1)
        numbers = re.findall(r"\d+", chunk)
        return [int(n) for n in numbers]
    return []


def compute_retrieval_f1(pred_keys: List[int], gold_keys: List[int]) -> float:
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


def load_reasonorderqa(dataset_path: Path, limit: int = None) -> List[Dict]:
    problems = []
    with open(dataset_path) as f:
        for line in f:
            problems.append(json.loads(line))
    if limit:
        problems = problems[:limit]
    return problems


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen on ReasonOrderQA via API")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_path", type=Path, default=Path("data/reasonorderqa/reasonorderqa.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--answer_first", action="store_true")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default="https://api.siliconflow.cn/v1")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    prompt_style = "answer_first" if args.answer_first else "cot_first"
    instruction = ANSWER_FIRST_INSTRUCTION if args.answer_first else COT_FIRST_INSTRUCTION

    run_name = args.run_name or f"qwen_reasonorderqa_{prompt_style}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / f"{run_name}.jsonl"
    summary_json = args.output_dir / f"{run_name}_summary.json"

    print(f"Loading ReasonOrderQA from {args.dataset_path}...")
    problems = load_reasonorderqa(args.dataset_path, args.limit)
    print(f"Evaluating {len(problems)} problems ({prompt_style})")

    batcher = GPTBatcher(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    prompts = []
    for problem in problems:
        question = problem["question"]
        prompt = f"{question}\n\n{instruction}"
        prompts.append(prompt)

    print("Running batch inference...")
    completions = batcher.batch_complete(
        prompts,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        batch_size=args.batch_size,
    )

    correct = 0
    total_f1 = 0.0
    results_by_difficulty = defaultdict(list)

    with open(output_jsonl, "w") as writer:
        for idx, (problem, completion) in enumerate(zip(problems, completions)):
            gold_answer = str(problem["answer"])
            metadata = problem.get("metadata", {})
            gold_keys = metadata.get("gold_keys", [])
            difficulty = metadata.get("difficulty", 0)

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
                "retrieval_f1": retrieval_f1,
                "completion": completion,
            }
            results_by_difficulty[difficulty].append(record)
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{idx+1}/{len(problems)}] D{difficulty} pred={pred_answer} gold={gold_answer} correct={is_correct}")

    accuracy = correct / len(problems)
    mean_f1 = total_f1 / len(problems)

    difficulty_stats = {}
    for d in [1, 2, 3, 4]:
        if results_by_difficulty[d]:
            d_correct = sum(1 for r in results_by_difficulty[d] if r["correct"])
            d_total = len(results_by_difficulty[d])
            difficulty_stats[f"D{d}_accuracy"] = d_correct / d_total
            difficulty_stats[f"D{d}_count"] = d_total

    summary = {
        "run_name": run_name,
        "model": args.model,
        "accuracy": accuracy,
        "mean_retrieval_f1": mean_f1,
        "correct": correct,
        "total": len(problems),
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
