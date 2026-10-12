#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("sequential_method", "Sequential", "#8c8c8c", ""),
    ("existing_method", "Existing", "#4c78a8", "//"),
    ("existing_method_green_context", "Existing + GC", "#f2cf5b", "xx"),
    ("proposed_method", "Proposed", "#e45756", ".."),
)

LABELS = {
    "vector_square": "Vector\nSquare",
    "black_scholes": "Black–\nScholes",
    "machine_learning": "Machine\nLearning",
    "hits": "HITS",
    "image_processing": "Image\nProcessing",
    "deep_learning": "Deep\nLearning",
    "synthetic_fully_parallel": "Fully\nParallel",
    "synthetic_mixed_chain_parallel": "Mixed Chain\n/ Parallel",
    "synthetic_multiple_long_branches": "Multiple Long\nBranches",
    "synthetic_random_dag": "Random\nDAG",
    "synthetic_sequential_chain": "Sequential\nChain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--group", choices=("kesco", "stg", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values: dict[tuple[str, str], float] = {}
    order: list[str] = []
    with args.csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            benchmark = row["benchmark"]
            if benchmark not in order:
                order.append(benchmark)
            values[(benchmark, row["method"])] = float(row["sm_active_avg_pct"])

    if args.group == "kesco":
        order = [name for name in order if not name.startswith("synthetic_")]
        title = "SM Utilization — Real CUDA Kernels"
    elif args.group == "stg":
        order = [name for name in order if name.startswith("synthetic_")]
        title = "SM Utilization — Synthetic STG Kernels"
    else:
        title = "SM Utilization"

    x = np.arange(len(order))
    width = 0.19
    figure, axis = plt.subplots(figsize=(max(11, len(order) * 2.0), 7.2))

    for index, (method, label, color, hatch) in enumerate(METHODS):
        heights = [values[(benchmark, method)] for benchmark in order]
        positions = x + (index - 1.5) * width
        bars = axis.bar(
            positions,
            heights,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            hatch=hatch,
        )
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in heights], padding=3, fontsize=8, rotation=90)

    axis.set_title(title, fontsize=18, pad=16)
    axis.set_ylabel("SMs Active (%)", fontsize=14)
    axis.set_xticks(x, [LABELS.get(name, name) for name in order], fontsize=11)
    axis.set_ylim(0, 110)
    axis.set_yticks(np.arange(0, 101, 10))
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), fontsize=11)
    figure.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"graph written: {args.output_path}")


if __name__ == "__main__":
    main()
