#!/usr/bin/env python3
"""Measure Nsight Systems SMs Active for every benchmark/method pair."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sqlite3
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parents[2]
BENCHMARK_DIR = STUDY_DIR / "benchmarks"

METHODS = (
    "sequential_method",
    "existing_method",
    "existing_method_green_context",
    "proposed_method",
)

BENCHMARKS = (
    "vector_square",
    "black_scholes",
    "machine_learning",
    "hits",
    "image_processing",
    "deep_learning",
    "synthetic_fully_parallel",
    "synthetic_mixed_chain_parallel",
    "synthetic_multiple_long_branches",
    "synthetic_random_dag",
    "synthetic_sequential_chain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs="*")
    parser.add_argument("--frequency", type=int, default=10_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STUDY_DIR / "results/benchmark_evaluation/latest-sm-active",
    )
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=STUDY_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{tail}"
        )


def query_sm_active(sqlite_path: Path) -> dict[str, float | int]:
    connection = sqlite3.connect(sqlite_path)
    try:
        kernels = connection.execute(
            """
            SELECT k.start, k.end, names.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS names ON names.id = k.demangledName
            ORDER BY k.start
            """
        ).fetchall()

        # Workspace initialization is outside gpu_submit_wait_ms and must not
        # contaminate the task-execution utilization window.
        task_kernels = [
            row for row in kernels
            if "initialize_array" not in row[2]
            and "initialize_csr" not in row[2]
        ]
        if not task_kernels:
            raise RuntimeError(f"task kernels were not found in {sqlite_path}")

        window_start = min(int(row[0]) for row in task_kernels)
        window_end = max(int(row[1]) for row in task_kernels)
        samples = connection.execute(
            """
            SELECT metrics.value
            FROM GPU_METRICS AS metrics
            JOIN TARGET_INFO_GPU_METRICS AS info USING (metricId)
            WHERE info.metricName LIKE 'SMs Active%'
              AND metrics.timestamp BETWEEN ? AND ?
            ORDER BY metrics.timestamp
            """,
            (window_start, window_end),
        ).fetchall()
    finally:
        connection.close()

    values = [float(row[0]) for row in samples]
    if not values:
        raise RuntimeError(f"SMs Active samples were not found in {sqlite_path}")

    return {
        "average": sum(values) / len(values),
        "minimum": min(values),
        "maximum": max(values),
        "samples": len(values),
        "window_ms": (window_end - window_start) / 1_000_000.0,
        "kernel_count": len(task_kernels),
    }


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.benchmarks) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"unknown benchmarks: {', '.join(unknown)}")
    selected = tuple(args.benchmarks) if args.benchmarks else BENCHMARKS
    output_dir = args.output_dir.resolve()
    report_dir = output_dir / "reports"
    log_dir = output_dir / "logs"
    tmp_dir = output_dir / "tmp"
    for directory in (output_dir, report_dir, log_dir, tmp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_dir)
    env["STG_DISABLE_STREAM_PLOT"] = "1"

    csv_path = output_dir / "sm_active_results.csv"
    fieldnames = (
        "benchmark", "method", "sm_active_avg_pct", "sm_active_min_pct",
        "sm_active_max_pct", "samples", "window_ms", "kernel_count",
        "nsys_report", "sqlite",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for benchmark in selected:
            group = "STG" if benchmark.startswith("synthetic_") else "KESCO"
            stg = BENCHMARK_DIR / group / f"BENCHMARK_{benchmark}" / f"{benchmark}.stg"
            for method in METHODS:
                executable = STUDY_DIR / method / "main"
                stem = f"{benchmark}_{method}"
                prefix = report_dir / stem
                report = prefix.with_suffix(".nsys-rep")
                sqlite_path = prefix.with_suffix(".sqlite")
                profile_log = log_dir / f"{stem}_profile.log"
                export_log = log_dir / f"{stem}_export.log"

                print(f"[SMs Active] {benchmark} / {method}", flush=True)
                run_logged(
                    [
                        "nsys", "profile", "--force-overwrite=true",
                        "--sample=none", "--cpuctxsw=none", "--trace=cuda",
                        "--gpu-metrics-devices=0",
                        f"--gpu-metrics-frequency={args.frequency}",
                        f"--output={prefix}", str(executable), str(stg),
                    ],
                    profile_log,
                    env,
                )
                run_logged(
                    [
                        "nsys", "export", "--type=sqlite",
                        "--force-overwrite=true", f"--output={sqlite_path}",
                        str(report),
                    ],
                    export_log,
                    env,
                )
                result = query_sm_active(sqlite_path)
                print(
                    f"  avg={result['average']:.2f}% "
                    f"min={result['minimum']:.2f}% max={result['maximum']:.2f}% "
                    f"samples={result['samples']} window={result['window_ms']:.3f} ms",
                    flush=True,
                )
                writer.writerow(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "sm_active_avg_pct": f"{result['average']:.6f}",
                        "sm_active_min_pct": f"{result['minimum']:.6f}",
                        "sm_active_max_pct": f"{result['maximum']:.6f}",
                        "samples": result["samples"],
                        "window_ms": f"{result['window_ms']:.6f}",
                        "kernel_count": result["kernel_count"],
                        "nsys_report": report,
                        "sqlite": sqlite_path,
                    }
                )
                stream.flush()

    print(f"result: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
