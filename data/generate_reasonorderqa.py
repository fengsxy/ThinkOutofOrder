#!/usr/bin/env python3
"""
Generate ReasonOrderQA dataset for order robustness evaluation.

This synthetic dataset tests retrieval + reasoning capabilities by embedding
secret numbers in distractor text and requiring arithmetic computation.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Difficulty distribution
DIFFICULTY_DIST = {
    1: 0.25,  # Simple: X + Y + Z
    2: 0.40,  # Medium: X + Y - Z or X - Y + Z
    3: 0.25,  # Hard: (X + Y) * Z or (X - Y) * Z
    4: 0.10,  # Very Hard: (X - Y * Z) * W
}

# Expression templates by difficulty
EXPRESSIONS = {
    1: ["X + Y + Z"],
    2: ["X + Y - Z", "X - Y + Z", "X * Y + Z"],
    3: ["(X + Y) * Z", "(X - Y) * Z", "X * Y - Z"],
    4: ["(X - Y * Z) * W", "(X + Y) * Z - W", "X * Y + Z * W"],
}

# Distractor sentences
DISTRACTORS = [
    "The annual tech conference was canceled unexpectedly due to budget issues.",
    "Most attendees had already booked their flights.",
    "Global weather patterns suggest a mild winter this year.",
    "Archaeologists discovered new artifacts near the old trade route.",
    "The policy committee will review the draft proposal next month.",
    "Heavy rainfall caused delays in several construction projects.",
    "A survey indicated that smartphone usage continued to rise.",
    "The research team plans to publish a follow-up study next March.",
    "Local markets reported an unusual spike in afternoon visitors.",
    "Another report highlighted concerns about long-term water supplies.",
    "Many volunteers stayed late to complete the annual cleanup.",
    "Analysts debated whether the growth trend would continue next quarter.",
]

FILLER_FRAGMENTS = [
    "The outlook remains cautiously optimistic according to early notes.",
    "Further details are expected in the upcoming quarterly review.",
    "Preliminary drafts suggested a need for broader collaboration.",
    "Stakeholders emphasized clarity, pacing, and measurable outcomes.",
    "The interim summary captured both setbacks and small wins.",
    "Several short-term fixes were prototyped to gather feedback.",
    "Teams documented their assumptions to avoid hidden coupling.",
    "Observers noted that attention to detail improved reliability.",
]

PROMPT_HEADER = """You will read a long passage that contains many irrelevant sentences.
Somewhere in the passage, several SECRET NUMBERS are hidden in the form:

"The secret key X is <number>."
"The secret key Y is <number>."
"The secret key Z is <number>."

Your task has THREE PHASES, and you MUST follow the output format strictly.
Expression to evaluate: {expr}

========================
### PASSAGE STARTS
"""

PROMPT_SUFFIX = """
### PASSAGE ENDS
========================
"""


def sample_difficulty() -> int:
    """Sample difficulty level according to distribution."""
    r = random.random()
    cumulative = 0.0
    for level, prob in DIFFICULTY_DIST.items():
        cumulative += prob
        if r < cumulative:
            return level
    return max(DIFFICULTY_DIST.keys())


def generate_passage(
    keys: List[Tuple[str, int]],
    target_length: int = 1000,
) -> Tuple[str, int]:
    """Generate passage with embedded secret keys."""
    key_lines = [f"The secret key {name} is {value}." for name, value in keys]

    # Generate distractor lines
    num_distractors = max(8, target_length // 100)
    distractor_lines = []
    for _ in range(num_distractors):
        d1 = random.choice(DISTRACTORS)
        d2 = random.choice(FILLER_FRAGMENTS)
        distractor_lines.append(f"{d1} {d2}")

    # Combine and shuffle
    all_lines = key_lines + distractor_lines
    random.shuffle(all_lines)

    passage = "\n".join(all_lines)
    return passage, len(passage)


def evaluate_expression(expr: str, values: Dict[str, int]) -> int:
    """Safely evaluate arithmetic expression."""
    # Replace variable names with values
    for var, val in values.items():
        expr = expr.replace(var, str(val))
    return eval(expr)


def generate_problem(
    target_length: int = 1000,
    difficulty: Optional[int] = None,
) -> Dict:
    """Generate a single ReasonOrderQA problem."""
    if difficulty is None:
        difficulty = sample_difficulty()

    # Select expression template
    expr_template = random.choice(EXPRESSIONS[difficulty])

    # Determine required variables
    var_names = []
    for var in ["X", "Y", "Z", "W"]:
        if var in expr_template:
            var_names.append(var)

    # Generate random values
    values = {}
    keys = []
    for var in var_names:
        val = random.randint(1, 100)
        values[var] = val
        keys.append((var, val))

    # Compute answer
    answer = evaluate_expression(expr_template, values)

    # Generate passage
    passage, actual_length = generate_passage(keys, target_length)

    # Build question
    question = PROMPT_HEADER.format(expr=expr_template) + passage + PROMPT_SUFFIX

    return {
        "question": question,
        "answer": str(answer),
        "metadata": {
            "gold_keys": [v for _, v in keys],
            "expr": expr_template,
            "difficulty": difficulty,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate ReasonOrderQA dataset")
    parser.add_argument("--output", type=Path, default=Path("data/reasonorderqa/reasonorderqa.jsonl"))
    parser.add_argument("--count", type=int, default=1000, help="Number of problems to generate")
    parser.add_argument("--target_length", type=int, default=1000, help="Target passage length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    problems = []
    for _ in range(args.count):
        problems.append(generate_problem(target_length=args.target_length))

    with open(args.output, "w") as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")

    # Print statistics
    difficulties = [p["metadata"]["difficulty"] for p in problems]
    print(f"Generated {args.count} problems to {args.output}")
    print(f"Difficulty distribution: {dict(sorted([(d, difficulties.count(d)) for d in set(difficulties)]))}")


if __name__ == "__main__":
    main()
