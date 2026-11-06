#!/usr/bin/env python3
"""Create figures that directly support claims in the research summary."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/evidence_figures"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)


def synthetic_comparison():
    rows = read_csv(ROOT / "results/method_comparison_figures/RESULT_stg/all_stg_results.csv")
    benchmarks = [
        "synthetic_fully_parallel",
        "synthetic_mixed_chain_parallel",
        "synthetic_multiple_long_branches",
        "synthetic_random_dag",
        "synthetic_sequential_chain",
    ]
    methods = [
        ("sequential_method", "Sequential"),
        ("existing_method", "Existing"),
        ("existing_method_green_context", "Existing + GC"),
        ("proposed_method", "Proposed"),
    ]
    means = {
        (row["benchmark"], row["method"]): float(row["mean_gpu_kernel_ms"])
        for row in rows
    }
    ci = {
        (row["benchmark"], row["method"]): float(row["ci95_gpu_kernel_ms"])
        for row in rows
    }
    labels = [
        "Fully\nparallel",
        "Mixed chain\n+ parallel",
        "Multiple long\nbranches",
        "Random DAG",
        "Sequential\nchain",
    ]
    fig, axis = plt.subplots(figsize=(12, 6.5))
    x = list(range(len(benchmarks)))
    width = 0.19
    for index, (method, label) in enumerate(methods):
        offset = (index - 1.5) * width
        axis.bar(
            [value + offset for value in x],
            [means[(benchmark, method)] for benchmark in benchmarks],
            width,
            yerr=[ci[(benchmark, method)] for benchmark in benchmarks],
            capsize=3,
            label=label,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("gpu_kernel_ms (ms)")
    axis.set_title("Synthetic STG method comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save(fig, "synthetic_method_comparison.png")


def sm_utilization():
    rows = read_csv(ROOT / "results/method_comparison_figures/sm_utilization/sm_active_results.csv")
    benchmarks = ["vector_square", "black_scholes", "machine_learning", "hits", "image_processing", "deep_learning"]
    methods = [
        ("sequential_method", "Sequential"),
        ("existing_method", "Existing"),
        ("existing_method_green_context", "Existing + GC"),
        ("proposed_method", "Proposed"),
    ]
    values = {
        (row["benchmark"], row["method"]): float(row["sm_active_avg_pct"])
        for row in rows
    }
    fig, axis = plt.subplots(figsize=(12, 6.5))
    x = list(range(len(benchmarks)))
    width = 0.19
    for index, (method, label) in enumerate(methods):
        axis.bar(
            [value + (index - 1.5) * width for value in x],
            [values[(benchmark, method)] for benchmark in benchmarks],
            width,
            label=label,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(benchmarks, rotation=25, ha="right")
    axis.set_ylabel("SMs Active (%)")
    axis.set_title("Average SM activity")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save(fig, "sm_utilization_comparison.png")


def application_comparison():
    rows = read_csv(
        ROOT / "results/application_ablation_corrected/application_ablation_results.csv"
    )
    benchmarks = [
        "vector_square",
        "black_scholes",
        "machine_learning",
        "hits",
        "image_processing",
        "deep_learning",
    ]
    methods = [
        ("full", "Proposed"),
        ("no_importance", "No importance"),
        ("no_sm_allocation", "No SM allocation"),
        ("no_stream_placement", "No Stream placement"),
    ]
    means = {
        (row["benchmark"], row["variant"]):
        float(row["mean_gpu_kernel_ms"])
        for row in rows
    }
    ci = {
        (row["benchmark"], row["variant"]):
        float(row["ci95_gpu_kernel_ms"])
        for row in rows
    }
    labels = [
        "Vector\nsquare",
        "Black-\nScholes",
        "Machine\nlearning",
        "HITS",
        "Image\nprocessing",
        "Deep\nlearning",
    ]
    fig, axis = plt.subplots(figsize=(12, 6.5))
    x = list(range(len(benchmarks)))
    width = 0.19
    for index, (variant, label) in enumerate(methods):
        offset = (index - 1.5) * width
        axis.bar(
            [value + offset for value in x],
            [means[(benchmark, variant)] for benchmark in benchmarks],
            width,
            yerr=[ci[(benchmark, variant)] for benchmark in benchmarks],
            capsize=3,
            label=label,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("gpu_kernel_ms (ms)")
    axis.set_title("Real application method comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save(fig, "application_method_comparison.png")


def real_application_four_methods():
    rows = read_csv(ROOT / "results/benchmark_evaluation/latest-all/all_results.csv")
    benchmarks = [
        "vector_square",
        "black_scholes",
        "machine_learning",
        "hits",
        "image_processing",
        "deep_learning",
    ]
    methods = [
        ("sequential_method", "Sequential"),
        ("existing_method", "Existing"),
        ("existing_method_green_context", "Existing + GC"),
        ("proposed_method", "Proposed"),
    ]
    means = {
        (row["benchmark"], row["method"]):
        float(row["mean_gpu_kernel_ms"])
        for row in rows
        if row["benchmark"] in benchmarks
    }
    ci = {
        (row["benchmark"], row["method"]):
        float(row["ci95_gpu_kernel_ms"])
        for row in rows
        if row["benchmark"] in benchmarks
    }
    labels = [
        "Vector\nsquare",
        "Black-\nScholes",
        "Machine\nlearning",
        "HITS",
        "Image\nprocessing",
        "Deep\nlearning",
    ]
    fig, axis = plt.subplots(figsize=(12, 6.5))
    x = list(range(len(benchmarks)))
    width = 0.19
    for index, (method, label) in enumerate(methods):
        offset = (index - 1.5) * width
        axis.bar(
            [value + offset for value in x],
            [means[(benchmark, method)] for benchmark in benchmarks],
            width,
            yerr=[ci[(benchmark, method)] for benchmark in benchmarks],
            capsize=3,
            label=label,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("gpu_kernel_ms (ms)")
    axis.set_title("Real application comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save(fig, "real_application_four_methods.png")


def stream_candidate():
    path = ROOT / "results/benchmark_evaluation/latest-all/proposed_method/work/stream_plots/synthetic_mixed_chain_parallel_stream_makespan.csv"
    rows = read_csv(path)
    fig, axis = plt.subplots(figsize=(8, 5.5))
    stream_counts = [row["stream_count"] for row in rows]
    makespans = [float(row["estimated_makespan"]) for row in rows]
    colors = ["#d62728" if row["selected"] == "1" else "#7f7f7f" for row in rows]
    axis.bar(stream_counts, makespans, color=colors)
    axis.set_xlabel("Stream count")
    axis.set_ylabel("Estimated makespan")
    axis.set_title("SM allocation candidates: Mixed Chain + Parallel")
    axis.grid(axis="y", alpha=0.25)
    save(fig, "stream_candidate_makespan.png")


if __name__ == "__main__":
    synthetic_comparison()
    application_comparison()
    real_application_four_methods()
    sm_utilization()
    stream_candidate()
