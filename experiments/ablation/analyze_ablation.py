#!/usr/bin/env python3
"""Summarize staged ablation results without third-party dependencies."""

import csv
import sys
from pathlib import Path


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: analyze_ablation.py RESULTS SUMMARY_CSV SUMMARY_MD")

    input_path, summary_path, markdown_path = map(Path, sys.argv[1:])
    rows = read_rows(input_path)

    baseline = {
        row["benchmark"]: float(row["mean_gpu_kernel_ms"])
        for row in rows
        if row["status"] == "ok"
        and row["variant"] == "full"
    }

    summaries = []
    for row in sorted(rows, key=lambda item: (item["variant"], item["benchmark"])):
        if row["status"] != "ok":
            continue
        variant = row["variant"]
        benchmark = row["benchmark"]
        average = float(row["mean_gpu_kernel_ms"])
        ci95 = float(row["ci95_gpu_kernel_ms"])
        baseline_value = baseline.get(benchmark)
        speedup = (
            baseline_value / average
            if baseline_value is not None and average > 0
            else None
        )
        reduction = (
            (baseline_value - average) / baseline_value * 100.0
            if baseline_value
            else None
        )
        summaries.append({
            "variant": variant,
            "benchmark": benchmark,
            "mean_gpu_kernel_ms": f"{average:.6f}",
            "ci95_gpu_kernel_ms": f"{ci95:.6f}",
            "baseline_speedup": "" if speedup is None else f"{speedup:.6f}",
            "baseline_reduction_percent": (
                "" if reduction is None else f"{reduction:.6f}"
            ),
        })

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(summaries[0]) if summaries else [
            "variant", "benchmark", "mean_gpu_kernel_ms",
            "ci95_gpu_kernel_ms", "baseline_speedup",
            "baseline_reduction_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Ablation summary\n\n")
        handle.write(
            "Speedup and reduction are relative to the `full` proposed "
            "pipeline; lower time is better.\n\n"
        )
        handle.write(
            "| Variant | Benchmark | Mean (ms) | 95% CI (ms) | "
            "Speedup | Reduction |\n"
            "|---|---|---:|---:|---:|---:|\n"
        )
        for row in summaries:
            speedup = row["baseline_speedup"] or "n/a"
            reduction = row["baseline_reduction_percent"]
            reduction = "n/a" if not reduction else f"{reduction}%"
            handle.write(
                f"| {row['variant']} | {row['benchmark']} | "
                f"{row['mean_gpu_kernel_ms']} | "
                f"{row['ci95_gpu_kernel_ms']} | "
                f"{speedup} | {reduction} |\n"
            )


if __name__ == "__main__":
    main()
