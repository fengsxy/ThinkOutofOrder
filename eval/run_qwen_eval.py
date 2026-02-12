#!/usr/bin/env python3
"""
Evaluate Qwen autoregressive model on GSM8K using batch API calls.

This provides a baseline comparison for autoregressive models against
discrete diffusion models on order robustness.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpt_batch.batcher import GPTBatcher


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

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract numeric answer from model output."""
    if not text:
        return None
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        content = boxed_match.group(1).strip()
        num_match = re.search(r"-?\d[\d,]*\.?\d*", content.replace(",", ""))
        if num_match:
            return num_match.group(0)
    hash_match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if hash_match:
        return hash_match.group(1).replace(",", "")
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


def load_gsm8k(dataset_path: Path, split: str = "test", limit: int = None) -> List[Dict]:
    """Load GSM8K dataset."""
    jsonl_path = dataset_path / f"{split}.jsonl"
    problems = []
    with open(jsonl_path) as f:
        for line in f:
            problems.append(json.loads(line))
    if limit:
        problems = problems[:limit]
    return problems


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen on GSM8K via API")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_path", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--split", type=str, default="test")
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

    run_name = args.run_name or f"qwen_gsm8k_{prompt_style}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / f"{run_name}.jsonl"
    summary_json = args.output_dir / f"{run_name}_summary.json"

    print(f"Loading GSM8K from {args.dataset_path}...")
    problems = load_gsm8k(args.dataset_path, args.split, args.limit)
    print(f"Evaluating {len(problems)} problems ({prompt_style})")

    # Get API key from args or environment
    import os
    api_key = args.api_key or os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("API key required. Set --api_key or SILICONFLOW_API_KEY environment variable.")

    # Initialize batcher
    batcher = GPTBatcher(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Prepare prompts
    prompts = []
    for problem in problems:
        question = problem["question"]
        prompt = f"Question: {question}\n\n{instruction}"
        prompts.append(prompt)

    # Run batch inference
    print("Running batch inference...")
    completions = batcher.batch_complete(
        prompts,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        batch_size=args.batch_size,
    )

    # Evaluate results
    correct = 0
    results = []

    with open(output_jsonl, "w") as writer:
        for idx, (problem, completion) in enumerate(zip(problems, completions)):
            gold_answer = parse_gold_answer(problem["answer"])
            pred_answer = extract_numeric_answer(completion)

            is_correct = pred_answer is not None and str(pred_answer) == str(gold_answer)
            if is_correct:
                correct += 1

            record = {
                "idx": idx,
                "question": problem["question"],
                "gold": gold_answer,
                "pred": pred_answer,
                "correct": is_correct,
                "completion": completion,
            }
            results.append(record)
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{idx+1}/{len(problems)}] pred={pred_answer} gold={gold_answer} correct={is_correct}")

    accuracy = correct / len(problems)
    summary = {
        "run_name": run_name,
        "model": args.model,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "prompt_style": prompt_style,
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(problems)})")


if __name__ == "__main__":
    main()
