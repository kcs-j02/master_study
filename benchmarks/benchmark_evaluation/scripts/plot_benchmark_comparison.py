#!/usr/bin/env python3
"""Plot all successful benchmark/method rows from an evaluation CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BENCHMARKS = [
    ("vector_square", "Vector\nSquare"),
    ("black_scholes", "Black &\nScholes"),
    ("machine_learning", "Machine\nLearning"),
    ("hits", "HITS"),
    ("image_processing", "Image\nProcessing"),
    ("deep_learning", "Deep\nLearning"),
    ("synthetic_fully_parallel", "Synthetic\nFully Parallel"),
    ("synthetic_mixed_chain_parallel", "Synthetic Mixed\nChain + Parallel"),
    ("synthetic_multiple_long_branches", "Synthetic Multiple\nLong Branches"),
    ("synthetic_random_dag", "Synthetic\nRandom DAG"),
    ("synthetic_sequential_chain", "Synthetic\nSequential Chain"),
]

METHODS = [
    ("sequential_method", "SEQ", "#f2f2f2", ""),
    ("existing_method", "Existing", "#d9e9f6", "///"),
    ("existing_method_green_context", "Existing+GC", "#dcead5", "xxx"),
    ("proposed_method", "Proposed", "#efcccc", "..."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    return parser.parse_args()


def read_results(path: Path) -> dict[tuple[str, str], tuple[float, float, float]]:
    results: dict[tuple[str, str], tuple[float, float, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "ok":
                continue
            key = (row["benchmark"], row["method"])
            results[key] = (
                float(row["mean_gpu_submit_wait_ms"]),
                float(row["min_gpu_submit_wait_ms"]),
                float(row["max_gpu_submit_wait_ms"]),
            )
    return results


def value_label(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def main() -> None:
    args = parse_args()
    results = read_results(args.csv_path)
    selected_benchmarks = [
        item
        for item in BENCHMARKS
        if any(key[0] == item[0] for key in results)
    ]
    if not selected_benchmarks:
        raise SystemExit("no successful benchmark rows found")

    expected = [
        (benchmark, method)
        for benchmark, _ in selected_benchmarks
        for method, *_ in METHODS
    ]
    missing = [key for key in expected if key not in results]
    if missing:
        formatted = ", ".join(f"{benchmark}/{method}" for benchmark, method in missing)
        raise SystemExit(f"missing successful result rows: {formatted}")

    x = np.arange(len(selected_benchmarks), dtype=float)
    width = 0.19
    figure_width = max(10.0, 2.25 * len(selected_benchmarks))
    figure, axis = plt.subplots(figsize=(figure_width, 8))

    largest = 0.0
    for method_index, (method, legend, color, hatch) in enumerate(METHODS):
        means = np.array([results[(benchmark, method)][0] for benchmark, _ in selected_benchmarks])
        minima = np.array([results[(benchmark, method)][1] for benchmark, _ in selected_benchmarks])
        maxima = np.array([results[(benchmark, method)][2] for benchmark, _ in selected_benchmarks])
        largest = max(largest, float(maxima.max()))
        positions = x + (method_index - (len(METHODS) - 1) / 2.0) * width
        bars = axis.bar(
            positions,
            means,
            width,
            label=legend,
            color=color,
            edgecolor="black",
            linewidth=1.4,
            hatch=hatch,
            yerr=np.vstack((means - minima, maxima - means)),
            capsize=3,
            error_kw={"elinewidth": 1.0},
        )
        axis.bar_label(
            bars,
            labels=[value_label(value) for value in means],
            padding=5,
            fontsize=10,
        )

    if len(selected_benchmarks) == 1:
        title = (
            "Method Comparison: "
            f"{selected_benchmarks[0][1].replace(chr(10), ' ')}"
        )
    elif all(
        benchmark.startswith("synthetic_")
        for benchmark, _ in selected_benchmarks
    ):
        title = "Execution Time Comparison Across Synthetic STGs"
    elif all(
        not benchmark.startswith("synthetic_")
        for benchmark, _ in selected_benchmarks
    ):
        title = "Execution Time Comparison Across Other Benchmarks"
    else:
        title = "Execution Time Comparison Across All Benchmarks"
    axis.set_title(title, fontsize=24)
    axis.set_ylabel("GPU Submit Wait Time [ms]", fontsize=18)
    axis.set_xlabel("Benchmark", fontsize=18)
    axis.set_xticks(x, [label for _, label in selected_benchmarks], fontsize=11)
    axis.tick_params(axis="y", labelsize=12)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.set_ylim(0, largest * 1.18 if largest > 0 else 1)
    axis.legend(ncols=4, fontsize=14, loc="upper right")
    figure.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"graph written: {args.output_path}")


if __name__ == "__main__":
    main()
