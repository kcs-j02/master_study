#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import statistics
import sys

import matplotlib.pyplot as plt


LABELS = {
    "vector_square": "Vector Square",
    "black_scholes": "Black–Scholes",
    "machine_learning": "Machine Learning",
    "hits": "HITS",
    "image_processing": "Image Processing",
    "deep_learning": "Deep Learning",
}

COMPARISON_SM_COUNTS = (64, 56, 48, 40, 32, 24, 16, 8)


def predicted_slowdown(sm_count: int) -> float:
    """Reproduce the current Stage 4 block-pass prediction (64 SM = 1)."""
    element_count = 64 * 256

    def pass_count(effective_sms: int) -> int:
        elements_per_block = (element_count + effective_sms - 1) // effective_sms
        threads = ((elements_per_block + 31) // 32) * 32
        threads = max(32, min(256, threads))
        return (elements_per_block + threads - 1) // threads

    return pass_count(sm_count) / pass_count(64)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_benchmark_totals(rows: list[dict[str, str]], output: Path) -> None:
    totals = [row for row in rows if row["scope"] == "benchmark"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for axis, benchmark in zip(axes.flat, LABELS):
        selected = sorted(
            (row for row in totals if row["benchmark"] == benchmark),
            key=lambda row: int(row["sm_count"]),
        )
        x = [int(row["sm_count"]) for row in selected]
        y = [float(row["median_ms"]) for row in selected]
        axis.plot(x, y, marker="o", linewidth=2)
        axis.set_title(LABELS[benchmark])
        axis.set_ylabel("Median time [ms]")
        axis.grid(True, linestyle="--", alpha=0.35)
    for axis in axes[-1]:
        axis.set_xlabel("Allocated SMs")
    figure.suptitle("KESCO Input Processing Time vs. Allocated SM Count", fontsize=16)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_slowdowns(rows: list[dict[str, str]], output: Path) -> None:
    totals = [row for row in rows if row["scope"] == "benchmark"]
    figure, axis = plt.subplots(figsize=(12, 7))
    for benchmark in LABELS:
        selected = sorted(
            (row for row in totals if row["benchmark"] == benchmark),
            key=lambda row: int(row["sm_count"]),
        )
        axis.plot(
            [int(row["sm_count"]) for row in selected],
            [float(row["slowdown_vs_full_gpu"]) for row in selected],
            marker="o", linewidth=2, label=LABELS[benchmark],
        )
    axis.axhline(1.0, color="black", linewidth=1, linestyle=":")
    axis.set_xlabel("Allocated SMs")
    axis.set_ylabel("Slowdown vs. full GPU")
    axis.set_title("KESCO SM Scaling by Input")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(fontsize=20, ncols=2)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_task_curves(rows: list[dict[str, str]], output_dir: Path) -> None:
    task_rows = [row for row in rows if row["scope"] == "task"]
    for benchmark in LABELS:
        grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in task_rows:
            if row["benchmark"] == benchmark:
                grouped[(row["kernel"], int(row["sm_count"]))].append(
                    float(row["median_ms"])
                )
        kernels = sorted({kernel for kernel, _ in grouped})
        figure, axis = plt.subplots(figsize=(12, 7))
        for kernel in kernels:
            sm_counts = sorted(sm for name, sm in grouped if name == kernel)
            values = [statistics.median(grouped[(kernel, sm)]) for sm in sm_counts]
            axis.plot(sm_counts, values, marker="o", linewidth=1.8, label=kernel)
        axis.set_xlabel("Allocated SMs")
        axis.set_ylabel("Median kernel time [ms]")
        axis.set_title(f"{LABELS[benchmark]} — Kernel SM Scaling")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(fontsize=16, ncols=2)
        figure.tight_layout()
        figure.savefig(
            output_dir / f"{benchmark}_kernel_sm_scaling.png",
            dpi=200, bbox_inches="tight",
        )
        plt.close(figure)


def plot_prediction_vs_measurement(
    rows: list[dict[str, str]], output: Path
) -> None:
    totals = [row for row in rows if row["scope"] == "benchmark"]
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    for axis, benchmark in zip(axes.flat, LABELS):
        by_sm = {
            int(row["sm_count"]): float(row["median_ms"])
            for row in totals
            if row["benchmark"] == benchmark
        }
        reference_ms = by_sm[64]
        measured = [by_sm[sm] / reference_ms for sm in COMPARISON_SM_COUNTS]
        predicted = [predicted_slowdown(sm) for sm in COMPARISON_SM_COUNTS]

        axis.plot(
            COMPARISON_SM_COUNTS,
            predicted,
            marker="s",
            linewidth=2.2,
            linestyle="--",
            color="black",
            label="Old Stage 4 prediction",
        )
        axis.plot(
            COMPARISON_SM_COUNTS,
            measured,
            marker="o",
            linewidth=2.5,
            label="Measured",
        )
        axis.set_title(LABELS[benchmark], fontsize=15)
        axis.set_ylabel("Time ratio (64 SM = 1)", fontsize=12)
        axis.set_xticks(COMPARISON_SM_COUNTS)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(fontsize=14)

    axes.flat[0].invert_xaxis()
    for axis in axes[-1]:
        axis.set_xlabel("Allocated SMs (64 → 8)", fontsize=12)
    figure.suptitle(
        "Old Stage 4 Prediction vs. Measured SM Scaling",
        fontsize=20,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_prediction_vs_all_benchmarks(
    rows: list[dict[str, str]], output: Path
) -> None:
    totals = [row for row in rows if row["scope"] == "benchmark"]
    figure, axis = plt.subplots(figsize=(15, 9))

    axis.plot(
        COMPARISON_SM_COUNTS,
        [predicted_slowdown(sm) for sm in COMPARISON_SM_COUNTS],
        marker="s",
        linewidth=3.2,
        linestyle="--",
        color="black",
        label="Old Stage 4 prediction",
        zorder=10,
    )

    for benchmark in LABELS:
        by_sm = {
            int(row["sm_count"]): float(row["median_ms"])
            for row in totals
            if row["benchmark"] == benchmark
        }
        reference_ms = by_sm[64]
        axis.plot(
            COMPARISON_SM_COUNTS,
            [by_sm[sm] / reference_ms for sm in COMPARISON_SM_COUNTS],
            marker="o",
            linewidth=2.5,
            label=f"{LABELS[benchmark]} measured",
        )

    axis.set_xticks(COMPARISON_SM_COUNTS)
    axis.invert_xaxis()
    axis.set_xlabel("Allocated SMs (64 → 8)", fontsize=16)
    axis.set_ylabel("Time ratio (64 SM = 1)", fontsize=16)
    axis.set_title(
        "Old Stage 4 Prediction vs. All Measured Benchmarks",
        fontsize=20,
    )
    axis.tick_params(axis="both", labelsize=13)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(fontsize=16, ncols=2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_revised_prediction_vs_measurement(
    rows: list[dict[str, str]], output: Path
) -> None:
    totals = [row for row in rows if row["scope"] == "benchmark"]
    task_rows = [row for row in rows if row["scope"] == "task"]
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    for axis, benchmark in zip(axes.flat, LABELS):
        total_by_sm = {
            int(row["sm_count"]): float(row["median_ms"])
            for row in totals
            if row["benchmark"] == benchmark
        }
        kernel_task_ids: dict[str, set[int]] = defaultdict(set)
        kernel_times: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in task_rows:
            if row["benchmark"] != benchmark:
                continue
            kernel = row["kernel"]
            sm_count = int(row["sm_count"])
            kernel_task_ids[kernel].add(int(row["task_id"]))
            kernel_times[(kernel, sm_count)].append(float(row["median_ms"]))

        model_by_sm = {
            sm: sum(
                statistics.median(kernel_times[(kernel, sm)]) * len(task_ids)
                for kernel, task_ids in kernel_task_ids.items()
            )
            for sm in COMPARISON_SM_COUNTS
        }
        measured_reference = total_by_sm[64]
        model_reference = model_by_sm[64]

        axis.plot(
            COMPARISON_SM_COUNTS,
            [model_by_sm[sm] / model_reference for sm in COMPARISON_SM_COUNTS],
            marker="s",
            linewidth=2.5,
            linestyle="--",
            color="black",
            label="Revised prediction",
        )
        axis.plot(
            COMPARISON_SM_COUNTS,
            [total_by_sm[sm] / measured_reference for sm in COMPARISON_SM_COUNTS],
            marker="o",
            linewidth=2.5,
            label="Measured",
        )
        axis.set_title(LABELS[benchmark], fontsize=15)
        axis.set_ylabel("Time ratio (64 SM = 1)", fontsize=12)
        axis.set_xticks(COMPARISON_SM_COUNTS)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(fontsize=14)

    axes.flat[0].invert_xaxis()
    for axis in axes[-1]:
        axis.set_xlabel("Allocated SMs (64 → 8)", fontsize=12)
    figure.suptitle(
        "Revised Per-Kernel Prediction vs. Measured SM Scaling",
        fontsize=20,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} INPUT.csv OUTPUT_DIR")
    csv_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(csv_path)
    plot_benchmark_totals(rows, output_dir / "kesco_benchmark_sm_scaling.png")
    plot_slowdowns(rows, output_dir / "kesco_benchmark_sm_slowdown.png")
    plot_task_curves(rows, output_dir)
    plot_prediction_vs_measurement(
        rows,
        output_dir / "stage4_prediction_vs_measured.png",
    )
    plot_prediction_vs_all_benchmarks(
        rows,
        output_dir / "stage4_prediction_vs_all_benchmarks.png",
    )
    plot_revised_prediction_vs_measurement(
        rows,
        output_dir / "stage4_revised_prediction_vs_measured.png",
    )
    print(f"graphs written: {output_dir}")


if __name__ == "__main__":
    main()
