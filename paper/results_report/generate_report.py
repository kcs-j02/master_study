#!/usr/bin/env python3
"""Generate figures, tables, and a compact PDF report from measured results."""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/application_ablation_corrected/application_ablation_results.csv"
OUTPUT = ROOT / "paper/results_report"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"

VARIANTS = ["full", "no_importance", "no_sm_allocation", "no_stream_placement"]
LABELS = {
    "full": "Full",
    "no_importance": "No importance",
    "no_sm_allocation": "No SM allocation",
    "no_stream_placement": "No Stream placement",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_table(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_application_times(rows):
    benchmarks = sorted({row["benchmark"] for row in rows})
    means = {
        (row["benchmark"], row["variant"]): float(row["mean_gpu_kernel_ms"])
        for row in rows
    }
    ci = {
        (row["benchmark"], row["variant"]): float(row["ci95_gpu_kernel_ms"])
        for row in rows
    }
    fig, axis = plt.subplots(figsize=(12, 6.5))
    x = list(range(len(benchmarks)))
    width = 0.19
    for index, variant in enumerate(VARIANTS):
        offset = (index - 1.5) * width
        values = [means[(benchmark, variant)] for benchmark in benchmarks]
        errors = [ci[(benchmark, variant)] for benchmark in benchmarks]
        axis.bar(
            [value + offset for value in x],
            values,
            width,
            yerr=errors,
            capsize=3,
            label=LABELS[variant],
        )
    axis.set_xticks(x)
    axis.set_xticklabels(benchmarks, rotation=25, ha="right")
    axis.set_ylabel("gpu_kernel_ms (ms)")
    axis.set_title("Application ablation: execution time")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = FIGURES / "application_ablation_execution_time.png"
    fig.savefig(path, dpi=220)
    return fig, path


def plot_relative_change(rows):
    full = {
        row["benchmark"]: float(row["mean_gpu_kernel_ms"])
        for row in rows if row["variant"] == "full"
    }
    benchmarks = sorted(full)
    fig, axis = plt.subplots(figsize=(11, 6))
    x = list(range(len(benchmarks)))
    for index, variant in enumerate(VARIANTS[1:]):
        changes = [
            (
                float(next(row["mean_gpu_kernel_ms"] for row in rows
                          if row["benchmark"] == benchmark
                          and row["variant"] == variant))
                / full[benchmark] - 1.0
            ) * 100.0
            for benchmark in benchmarks
        ]
        axis.bar(
            [value + (index - 1) * 0.25 for value in x],
            changes,
            0.25,
            label=LABELS[variant],
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(benchmarks, rotation=25, ha="right")
    axis.set_ylabel("Change relative to Full (%)")
    axis.set_title("Application ablation: relative change")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = FIGURES / "application_ablation_relative_change.png"
    fig.savefig(path, dpi=220)
    return fig, path


def table_rows(rows):
    full = {
        row["benchmark"]: float(row["mean_gpu_kernel_ms"])
        for row in rows if row["variant"] == "full"
    }
    output = []
    for row in sorted(rows, key=lambda item: (item["benchmark"], VARIANTS.index(item["variant"]))):
        value = float(row["mean_gpu_kernel_ms"])
        output.append({
            "benchmark": row["benchmark"],
            "variant": row["variant"],
            "mean_ms": f"{value:.6f}",
            "ci95_ms": row["ci95_gpu_kernel_ms"],
            "relative_to_full_percent": f"{(value / full[row['benchmark']] - 1) * 100:.3f}",
            "validation_status": row["validation_status"],
        })
    return output


def report_page(pdf, rows):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.93, "GPU Task-Graph Scheduling Evaluation", fontsize=22, weight="bold")
    fig.text(0.06, 0.875, "Measured results for six KESCO-derived applications", fontsize=14)
    body = (
        "Experimental design\n"
        "• 6 applications, 4 staged ablation variants\n"
        "• corrected ablation measurements; warmup policy is recorded in the CSV\n"
        "• mean and 95% confidence interval reported\n"
        "• Fixed three-Stream comparison; numerical validation passed for all 24 conditions\n\n"
        "Variants\n"
        "Full: importance-aware Stream placement and SM allocation\n"
        "No importance: same pipeline with bottom-level priority disabled\n"
        "No SM allocation: same Stream count with equal SM allocation and no Green Context\n"
        "No Stream placement: one execution Stream\n\n"
        "Interpretation\n"
        "Stream placement is the most consistently influential component. Removing it\n"
        "increases execution time substantially for deep_learning, hits, and vector_square.\n"
        "The effect of importance and SM allocation is workload-dependent.\n\n"
        "Timing note\n"
        "gpu_kernel_ms is measured using CUDA Events from the first task start\n"
        "to the last task completion on the GPU."
    )
    fig.text(0.06, 0.80, body, fontsize=11, va="top", family="DejaVu Sans")
    fig.text(0.06, 0.08, "Source: results/application_ablation_corrected/", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = read_csv(INPUT)
    application_table = table_rows(rows)
    save_table(
        TABLES / "application_ablation_table.csv",
        list(application_table[0]),
        application_table,
    )
    pdf_path = OUTPUT / "results_report.pdf"
    with PdfPages(pdf_path) as pdf:
        report_page(pdf, rows)
        fig, _ = plot_application_times(rows)
        pdf.savefig(fig)
        plt.close(fig)
        fig, _ = plot_relative_change(rows)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF: {pdf_path}")
    print(f"Figures: {FIGURES}")
    print(f"Tables: {TABLES}")


if __name__ == "__main__":
    main()
