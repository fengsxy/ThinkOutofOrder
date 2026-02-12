#!/usr/bin/env python3
"""Unified evaluation script for Dream diffusion model on GSM8K, Math500, and ReasonOrderQA."""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))


# ============================================================================
# Constants and prompts
# ============================================================================

DEFAULT_PROMPT_PREFIX = (
    "You are a careful math tutor who solves problems accurately. "
    "Provide the correct answer and explain your reasoning."
)
DEFAULT_PROMPT_SUFFIX = "Work through the question deliberately and ensure the final answer is correct."

# Dataset-specific prompts
GSM8K_COT_INSTRUCTION = '''Explain the solution with a careful chain-of-thought before giving the final numeric result, and report the final answer inside \\boxed{}.
You MUST output your answer in the following exact structure:

Reasoning:
(explain how you combined the numbers to obtain the final result)

Answer:
\\boxed{number}'''

GSM8K_ANSWER_FIRST_INSTRUCTION = '''Begin by stating the final numeric answer inside \\boxed{} before giving the detailed chain-of-thought explanation.
You MUST start the response with the literal text "Answer:" on its own line (no text before it).
Immediately after the heading, output the numeric answer wrapped in \\boxed{}.
Then leave a blank line and provide the reasoning under a "Reasoning:" heading.

Answer:
\\boxed{number}

Reasoning:
(explain how you combined the numbers to obtain the final result)'''

MATH500_COT_INSTRUCTION = '''Explain the solution step by step with a clear and logical reasoning process.
Do NOT approximate unless explicitly required.

At the end, report the final answer inside \\boxed{}.
The final answer may be an integer, fraction, algebraic expression, interval, or set.

You MUST output in the following exact structure:

Reasoning:
(explain how you solved the problem)

Answer:
\\boxed{final_answer}'''

MATH500_ANSWER_FIRST_INSTRUCTION = '''Begin by stating the final answer inside \\boxed{} before giving the detailed reasoning.
You MUST start the response with the literal text "Answer:" on its own line (no text before it).
Immediately after the heading, output the final answer wrapped in \\boxed{} (it may be an integer, fraction, algebraic expression, interval, or set).
Then leave a blank line and provide the reasoning under a "Reasoning:" heading.

Answer:
\\boxed{final_answer}

Reasoning:
(explain how you solved the problem)'''

REASONORDERQA_COT_INSTRUCTION = '''### YOUR TASK

You MUST output your answer in the following exact structure:

Retrieval:
(list all secret key numbers you found, in order)

Reasoning:
(explain how you combined the numbers to obtain the final result)

Answer:
(the final num ONLY, no extra text)'''

REASONORDERQA_ANSWER_FIRST_INSTRUCTION = '''### YOUR TASK

You MUST output your answer in the following exact structure:

Answer:
(the final num ONLY, no extra text)
Reasoning:
(explain how you combined the numbers to obtain the final result)
Retrieval:
(list all secret key numbers you found, in order)'''


# ============================================================================
# Data loading utilities
# ============================================================================

def dataset_file(path: Path, split: str) -> Path:
    if path.is_file():
        return path
    candidate = path / f"{split}.jsonl"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not find {candidate}. Provide a JSONL file or dataset directory.")
    return candidate


def _normalize_problem_record(row: Dict[str, Any]) -> Dict[str, Any]:
    question = row.get("question") or row.get("problem") or row.get("prompt")
    answer = row.get("answer") or row.get("solution") or row.get("response")
    if question is None or answer is None:
        raise KeyError(f"Missing question/answer fields in record keys={list(row.keys())}")
    metadata = row.get("metadata") or {}
    return {"question": question, "answer": answer, "metadata": metadata}


def load_dataset_problems(
    dataset_type: str, path: Path, split: str, hf_dataset: Optional[str]
) -> List[Dict[str, Any]]:
    """Load problems from various dataset formats."""
    if hf_dataset:
        ds = load_dataset(hf_dataset, split=split)
        return [_normalize_problem_record(row) for row in ds]
    
    rows: List[Dict[str, Any]] = []
    data_file = dataset_file(path, split)
    with data_file.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(_normalize_problem_record(record))
    return rows


# ============================================================================
# Answer extraction utilities
# ============================================================================

def parse_gold_answer(answer_text: str) -> str:
    """Extract gold answer from solution text."""
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", answer_text)
    if boxed_match:
        return boxed_match.group(1).strip().replace(",", "")
    if "####" in answer_text:
        return answer_text.split("####", maxsplit=1)[-1].strip().replace(",", "")
    return answer_text.strip().replace(",", "")


def _last_number(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d+\.\d+|-?\d+", text)
    if matches:
        return matches[-1].lstrip("+")
    return None


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract numeric answer from completion text."""
    if not text:
        return None
    
    # Try boxed format first
    boxed_sections = re.findall(r"\\boxed\{([^}]*)\}", text)
    for section in boxed_sections:
        answer = _last_number(section)
        if answer is not None:
            return answer
        cleaned = section.strip()
        if cleaned:
            return cleaned
    
    return _last_number(text)


ANSWER_SECTION_PATTERN = re.compile(
    r"Answer:\s*(.+?)(?=\n(?:Reasoning|Retrieval)\b|$)", re.IGNORECASE | re.DOTALL
)


def extract_reasonorder_answer(text: str) -> Optional[str]:
    """Extract numeric answers from Answer/Reasoning/Retrieval formatted completions."""
    if not text:
        return None
    match = ANSWER_SECTION_PATTERN.search(text)
    if match:
        chunk = match.group(1).strip()
        number = _last_number(chunk)
        if number is not None:
            return number
        if chunk:
            first_line = chunk.splitlines()[0].strip()
            if first_line:
                return first_line
    return extract_numeric_answer(text)


def normalize_numeric(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = text.strip().replace(",", "")
    return cleaned if cleaned else None


# ============================================================================
# Prompt building
# ============================================================================

def get_instructions(dataset_type: str, answer_first: bool) -> str:
    """Get the appropriate instruction based on dataset type and order."""
    if dataset_type == "gsm8k":
        return GSM8K_ANSWER_FIRST_INSTRUCTION if answer_first else GSM8K_COT_INSTRUCTION
    elif dataset_type == "math500":
        return MATH500_ANSWER_FIRST_INSTRUCTION if answer_first else MATH500_COT_INSTRUCTION
    elif dataset_type == "reasonorderqa":
        return REASONORDERQA_ANSWER_FIRST_INSTRUCTION if answer_first else REASONORDERQA_COT_INSTRUCTION
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def build_prompt(
    question: str,
    prefix: Optional[str],
    suffix: Optional[str],
    extra_instruction: Optional[str],
) -> str:
    """Build the full prompt text."""
    sections: List[str] = []
    if prefix and prefix.strip():
        sections.append(prefix.strip())
    sections.append(f"Question: {question.strip()}")

    tail_parts = [part.strip() for part in (suffix, extra_instruction) if part and part.strip()]
    if tail_parts:
        sections.append("\n".join(tail_parts))

    return "\n\n".join(sections).strip()


# ============================================================================
# Dream model decoding
# ============================================================================

def dream_decode(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 256,
    steps: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    ar_mode: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Decode using Dream diffusion model.
    
    Args:
        model: Dream model
        tokenizer: Tokenizer
        prompt_text: The prompt text
        max_new_tokens: Maximum tokens to generate
        steps: Number of diffusion steps (or tokens for AR mode)
        temperature: Sampling temperature
        top_p: Top-p sampling
        alg: Diffusion algorithm ("entropy" for diffusion, or other modes)
        alg_temp: Algorithm-specific temperature
        ar_mode: If True, use autoregressive generation
    
    Returns:
        Tuple of (completion_text, step_records)
    """
    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs["input_ids"].to(device=model.device)
    attention_mask = inputs["attention_mask"].to(device=model.device)
    
    step_records = []
    
    if ar_mode:
        # Autoregressive mode: step-by-step generation
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=max_new_tokens,  # One step per token for AR
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            alg="origin",  # Use origin for more AR-like behavior
            alg_temp=0.,
        )
    else:
        # Diffusion mode: parallel generation
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
        )
    
    # Extract generation
    prompt_len = input_ids.shape[1]
    generated_ids = output.sequences[0][prompt_len:].tolist()
    completion = tokenizer.decode(generated_ids)
    
    # Clean up completion (remove EOS tokens)
    if tokenizer.eos_token:
        completion = completion.split(tokenizer.eos_token)[0]
    
    # Record step info
    step_records.append({
        "step": steps if not ar_mode else max_new_tokens,
        "decoded": completion,
        "mode": "ar" if ar_mode else "diffusion",
    })
    
    return completion, step_records


# ============================================================================
# Main evaluation logic
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Dream model on math datasets.")
    
    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        default="Dream-org/Dream-v0-Instruct-7B",
        help="HuggingFace model path or local path.",
    )
    
    # Dataset
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["gsm8k", "math500", "reasonorderqa"],
        required=True,
        help="Type of dataset to evaluate.",
    )
    parser.add_argument("--hf_dataset", type=str, default=None, help="Optional HF dataset repo.")
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=None,
        help="Path to dataset file or directory.",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split.")
    parser.add_argument("--limit", type=int, help="Limit number of problems.")
    
    # Generation mode
    parser.add_argument(
        "--ar_mode",
        action="store_true",
        help="Use autoregressive generation instead of diffusion.",
    )
    parser.add_argument(
        "--answer_first",
        action="store_true",
        help="Use answer-first prompting (default is COT-first).",
    )
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--steps", type=int, default=256, help="Diffusion steps.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--alg", type=str, default="maskgit_plus", help="Diffusion algorithm (maskgit_plus uses top1 confidence from MaskGIT).")
    parser.add_argument("--alg_temp", type=float, default=0.0, help="Algorithm temperature.")
    
    # Prompts
    parser.add_argument("--prompt_prefix", type=str, default=DEFAULT_PROMPT_PREFIX)
    parser.add_argument("--prompt_suffix", type=str, default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument("--append_instruction", type=str, default=None)
    
    # Output
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()


def get_default_dataset_path(dataset_type: str) -> Path:
    """Get default dataset path based on type."""
    if dataset_type == "gsm8k":
        return Path("data/gsm8k")
    elif dataset_type == "math500":
        return Path("data/math500/test.jsonl")
    elif dataset_type == "reasonorderqa":
        return Path("reasonorderqa.jsonl")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # Set default dataset path if not provided
    if args.dataset_path is None:
        args.dataset_path = get_default_dataset_path(args.dataset_type)
    
    # Get instruction based on dataset and order
    if args.append_instruction is None:
        args.append_instruction = get_instructions(args.dataset_type, args.answer_first)
    
    # Load dataset
    problems = load_dataset_problems(
        args.dataset_type, args.dataset_path, args.split, args.hf_dataset
    )
    if args.limit is not None:
        problems = problems[: args.limit]
    if not problems:
        raise SystemExit(f"No problems loaded from {args.dataset_path}")
    
    print(f"Loaded {len(problems)} {args.dataset_type} problems")
    
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading Dream model from {args.model_path}...")
    
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = model.to(device).eval()
    
    # Setup output paths
    mode_str = "ar" if args.ar_mode else "diffusion"
    order_str = "answer_first" if args.answer_first else "cot_first"
    run_name = args.run_name or f"dream_{args.dataset_type}_{mode_str}_{order_str}_{args.max_new_tokens}"
    
    output_jsonl = args.output_jsonl or Path("results/dream") / f"{run_name}.jsonl"
    summary_json = args.summary_json or Path("results/dream") / f"{run_name}_summary.json"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Choose answer extractor based on dataset
    if args.dataset_type == "reasonorderqa":
        extract_answer = extract_reasonorder_answer
    else:
        extract_answer = extract_numeric_answer
    
    correct = 0
    records: List[Dict[str, Any]] = []
    
    with output_jsonl.open("w", encoding="utf-8") as writer:
        for idx, problem in enumerate(problems):
            question = problem["question"]
            gold_text = problem["answer"]
            gold_answer = parse_gold_answer(gold_text)
            gold_norm = normalize_numeric(gold_answer)
            metadata = problem.get("metadata", {})
            
            # Build prompt
            prompt = build_prompt(
                question, args.prompt_prefix, args.prompt_suffix, args.append_instruction
            )
            
            # Generate
            completion, step_records = dream_decode(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                max_new_tokens=args.max_new_tokens,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.alg,
                alg_temp=args.alg_temp,
                ar_mode=args.ar_mode,
            )
            
            pred_answer = extract_answer(completion)
            pred_norm = normalize_numeric(pred_answer)
            is_correct = bool(gold_norm is not None and pred_norm == gold_norm)
            correct += int(is_correct)
            
            record = {
                "index": idx,
                "question": question,
                "prompt": prompt,
                "completion": completion,
                "pred_answer": pred_answer,
                "pred_answer_normalized": pred_norm,
                "gold_answer": gold_answer,
                "gold_answer_normalized": gold_norm,
                "is_correct": is_correct,
                "mode": mode_str,
                "order": order_str,
                "metadata": metadata,
            }
            records.append(record)
            
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()
            
            print(f"[{idx + 1}/{len(problems)}] pred={pred_answer} gold={gold_answer} correct={is_correct}")
    
    accuracy = correct / len(problems)
    
    summary = {
        "run_name": run_name,
        "model_path": args.model_path,
        "dataset_type": args.dataset_type,
        "dataset_path": str(args.dataset_path),
        "split": args.split,
        "num_examples": len(problems),
        "num_correct": correct,
        "accuracy": accuracy,
        "ar_mode": args.ar_mode,
        "answer_first": args.answer_first,
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "alg": args.alg,
        "alg_temp": args.alg_temp,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    print(f"Wrote per-problem outputs to {output_jsonl}")
    print(f"Wrote summary to {summary_json}")
    print(f"Accuracy: {accuracy:.3%} ({correct}/{len(problems)})")


if __name__ == "__main__":
    main()
