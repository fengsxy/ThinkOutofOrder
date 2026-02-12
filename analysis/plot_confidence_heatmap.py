#!/usr/bin/env python3
"""
Combined heatmap: D1 and D4 side by side with shared axes.
Shows confidence dynamics across diffusion steps for different difficulty levels.

Usage:
    python analysis/plot_confidence_heatmap.py --input results/confidence_trace.jsonl --output figures/heatmap.png
"""

import json
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from transformers import AutoTokenizer

# Font settings for paper
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 28,
    "axes.titlesize": 32,
    "axes.labelsize": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
})

ANSWER_RE = re.compile(r"Answer:\s*(.+?)(?=\n(?:Reasoning|Retrieval)\b|$)", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"-?\d+\.\d+|-?\d+")


def extract_answer_text(completion: str) -> str | None:
    if not completion:
        return None
    match = ANSWER_RE.search(completion)
    if match:
        chunk = match.group(1).strip()
        if not chunk:
            return None
        numbers = NUMBER_RE.findall(chunk)
        if numbers:
            return numbers[-1].lstrip("+")
        return chunk.splitlines()[0].strip()
    return None


def load_difficulty_map(dataset_path: Path) -> dict:
    mapping = {}
    with dataset_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            rec = json.loads(line)
            level = rec.get("level")
            if level is None:
                level = (rec.get("metadata") or {}).get("difficulty")
            if level is not None:
                mapping[idx] = int(level)
    return mapping


def compute_mean_matrices(input_jsonl: Path, dataset_jsonl: Path, tokenizer_path: str):
    """Compute mean confidence matrices for each difficulty level."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    diff_map = load_difficulty_map(dataset_jsonl)

    sums = {}
    counts = {}
    span_sums = {}
    span_counts = {}

    with input_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = rec.get("index", rec.get("idx"))
            difficulty = rec.get("difficulty")
            if difficulty is None and idx is not None and idx in diff_map:
                difficulty = diff_map[idx]
            if difficulty is None:
                continue
            difficulty = int(difficulty)

            mat = np.asarray(rec.get("confidence_matrix"), dtype=np.float32)
            if mat.ndim != 2 or mat.shape[0] < 256:
                continue
            mat = mat[:256, :256]
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

            sums.setdefault(difficulty, np.zeros_like(mat, dtype=np.float64))
            counts[difficulty] = counts.get(difficulty, 0) + 1
            sums[difficulty] += mat

            # Get answer span
            completion = rec.get("completion", "")
            answer_text = extract_answer_text(completion)
            if answer_text:
                completion_ids = tokenizer.encode(completion, add_special_tokens=False)[:256]
                answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
                if answer_ids:
                    for i in range(0, len(completion_ids) - len(answer_ids) + 1):
                        if completion_ids[i:i+len(answer_ids)] == answer_ids:
                            span_sums.setdefault(difficulty, [0, 0])
                            span_counts[difficulty] = span_counts.get(difficulty, 0) + 1
                            span_sums[difficulty][0] += i
                            span_sums[difficulty][1] += min(256, i + len(answer_ids))
                            break

    results = {}
    for diff in sorted(sums):
        mean_mat = (sums[diff] / max(counts[diff], 1)).astype(np.float32)
        span = None
        if diff in span_sums and span_counts.get(diff, 0) > 0:
            start = int(round(span_sums[diff][0] / span_counts[diff]))
            end = int(round(span_sums[diff][1] / span_counts[diff]))
            span = (max(0, min(255, start)), max(start + 1, min(256, end)))
        results[diff] = {"matrix": mean_mat, "span": span, "count": counts[diff]}

    return results


def plot_combined_heatmap(data, output_path: Path):
    """
    Create combined D1 + D4 heatmap with:
    - Top row: Zoomed-in answer region
    - Bottom row: Full heatmap
    - Shared y-axis within each column
    - Single colorbar on the right
    """
    d1_data = data.get(1)
    d4_data = data.get(4)

    if not d1_data or not d4_data:
        print("Missing D1 or D4 data!")
        return

    # Transpose for plotting (steps on x-axis, tokens on y-axis)
    mat_d1 = d1_data["matrix"].T
    mat_d4 = d4_data["matrix"].T
    span_d1 = d1_data["span"]
    span_d4 = d4_data["span"]

    vmin, vmax = 0, 1

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1, 1, 0.05],
        height_ratios=[1, 2.5],
        hspace=0.08,
        wspace=0.15,
    )

    ax_d1_zoom = fig.add_subplot(gs[0, 0])
    ax_d1_full = fig.add_subplot(gs[1, 0])
    ax_d4_zoom = fig.add_subplot(gs[0, 1], sharey=ax_d1_zoom)
    ax_d4_full = fig.add_subplot(gs[1, 1], sharey=ax_d1_full)
    cax = fig.add_subplot(gs[:, 2])

    pad = 10

    # D1 Column
    if span_d1:
        y_start = max(0, span_d1[0] - pad)
        y_end = min(mat_d1.shape[0], span_d1[1] + pad)
    else:
        y_start, y_end = 0, 30
    zoom_d1 = mat_d1[y_start:y_end, :]

    ax_d1_zoom.imshow(zoom_d1, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    if span_d1:
        rect = patches.Rectangle(
            (0, span_d1[0] - y_start), zoom_d1.shape[1], span_d1[1] - span_d1[0],
            linewidth=2.5, edgecolor="red", facecolor="none"
        )
        ax_d1_zoom.add_patch(rect)
    ax_d1_zoom.set_title("D1: $X + Y + Z$", fontweight='bold')
    ax_d1_zoom.set_xticks([])

    im_d1 = ax_d1_full.imshow(mat_d1, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    if span_d1:
        rect = patches.Rectangle(
            (0, span_d1[0]), mat_d1.shape[1], span_d1[1] - span_d1[0],
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax_d1_full.add_patch(rect)

    # D4 Column
    if span_d4:
        y_start_d4 = max(0, span_d4[0] - pad)
        y_end_d4 = min(mat_d4.shape[0], span_d4[1] + pad)
    else:
        y_start_d4, y_end_d4 = 0, 30
    zoom_d4 = mat_d4[y_start_d4:y_end_d4, :]

    ax_d4_zoom.imshow(zoom_d4, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    if span_d4:
        rect = patches.Rectangle(
            (0, span_d4[0] - y_start_d4), zoom_d4.shape[1], span_d4[1] - span_d4[0],
            linewidth=2.5, edgecolor="red", facecolor="none"
        )
        ax_d4_zoom.add_patch(rect)
    ax_d4_zoom.set_title("D4: $(X - Y \\times Z) \\times W$", fontweight='bold')
    ax_d4_zoom.set_xticks([])
    plt.setp(ax_d4_zoom.get_yticklabels(), visible=False)

    im_d4 = ax_d4_full.imshow(mat_d4, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    if span_d4:
        rect = patches.Rectangle(
            (0, span_d4[0]), mat_d4.shape[1], span_d4[1] - span_d4[0],
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax_d4_full.add_patch(rect)
    plt.setp(ax_d4_full.get_yticklabels(), visible=False)

    cbar = fig.colorbar(im_d4, cax=cax)
    cbar.set_label("Confidence", fontsize=24)
    cbar.ax.tick_params(labelsize=20)

    fig.text(0.45, 0.02, "Diffusion Step", ha='center', fontsize=28)
    fig.text(0.02, 0.5, "Token Position", va='center', rotation='vertical', fontsize=28)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Save metadata
    meta = {
        "D1": {"span": span_d1, "count": d1_data["count"]},
        "D4": {"span": span_d4, "count": d4_data["count"]},
    }
    meta_path = output_path.with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot confidence heatmap")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with confidence matrices")
    parser.add_argument("--dataset", type=Path, default=Path("data/reasonorderqa/reasonorderqa.jsonl"))
    parser.add_argument("--tokenizer", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("figures/figure_heatmap_d1_d4_combined.png"))
    return parser.parse_args()


def main():
    args = parse_args()

    print("Computing mean matrices by difficulty...")
    data = compute_mean_matrices(args.input, args.dataset, args.tokenizer)

    print(f"Found difficulties: {list(data.keys())}")
    for diff, info in data.items():
        print(f"  D{diff}: {info['count']} samples, span={info['span']}")

    print("\nPlotting combined heatmap...")
    plot_combined_heatmap(data, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
