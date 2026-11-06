#!/usr/bin/env python3
"""Summarize scheduling overhead from strict-ablation execution logs."""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


VALUE = re.compile(
    r"^(stage_[1-4]_[a-z_]+|gpu_submit_wait_ms|gpu_kernel_ms):\s+"
    r"([0-9.eE+-]+)"
)


def parse_log(path):
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = VALUE.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    required = {
        "stage_1_stg_analysis_seconds",
        "stage_2_task_importance_seconds",
        "stage_3_stream_placement_seconds",
        "stage_4_sm_allocation_seconds",
        "gpu_submit_wait_ms",
        "gpu_kernel_ms",
    }
    missing = required - values.keys()
    if missing:
        raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")
    scheduling_ms = sum(
        values[f"stage_{stage}_{name}_seconds"] * 1000.0
        for stage, name in (
            (1, "stg_analysis"),
            (2, "task_importance"),
            (3, "stream_placement"),
            (4, "sm_allocation"),
        )
    )
    values["scheduling_overhead_ms"] = scheduling_ms
    values["overhead_to_gpu_kernel_percent"] = (
        scheduling_ms / values["gpu_kernel_ms"] * 100.0
        if values["gpu_kernel_ms"] > 0 else 0.0
    )
    values["overhead_to_submit_percent"] = (
        scheduling_ms / values["gpu_submit_wait_ms"] * 100.0
        if values["gpu_submit_wait_ms"] > 0 else 0.0
    )
    return values


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: measure_overhead.py RAW_ROOT PER_RUN_CSV SUMMARY_CSV"
        )
    raw_root, per_run_path, summary_path = map(Path, sys.argv[1:])
    per_run = []
    grouped = defaultdict(list)
    for path in sorted(raw_root.glob("full/logs/*.log")):
        values = parse_log(path)
        benchmark = path.name.rsplit("_run", 1)[0]
        run_number = int(path.stem.rsplit("run", 1)[1])
        if run_number <= 3:
            continue
        values.update({"benchmark": benchmark, "run": run_number})
        per_run.append(values)
        grouped[benchmark].append(values)

    if not per_run:
        raise SystemExit(f"no post-warmup full logs found below {raw_root}")

    fields = [
        "benchmark", "run",         "stage_1_stg_analysis_seconds", "stage_2_task_importance_seconds",
        "stage_3_stream_placement_seconds", "stage_4_sm_allocation_seconds",
        "scheduling_overhead_ms", "gpu_submit_wait_ms", "gpu_kernel_ms",
        "overhead_to_gpu_kernel_percent", "overhead_to_submit_percent",
    ]
    with per_run_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_run:
            writer.writerow({field: row[field] for field in fields})

    summary_fields = [
        "benchmark", "samples", "mean_scheduling_overhead_ms",
        "mean_gpu_submit_wait_ms", "mean_gpu_kernel_ms",
        "overhead_to_gpu_kernel_percent", "overhead_to_submit_percent",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for benchmark, rows in sorted(grouped.items()):
            average = lambda key: sum(row[key] for row in rows) / len(rows)
            overhead = average("scheduling_overhead_ms")
            kernel = average("gpu_kernel_ms")
            submit = average("gpu_submit_wait_ms")
            writer.writerow({
                "benchmark": benchmark,
                "samples": len(rows),
                "mean_scheduling_overhead_ms": f"{overhead:.9f}",
                "mean_gpu_submit_wait_ms": f"{submit:.9f}",
                "mean_gpu_kernel_ms": f"{kernel:.9f}",
                "overhead_to_gpu_kernel_percent": (
                    f"{overhead / kernel * 100.0:.6f}" if kernel else "0"
                ),
                "overhead_to_submit_percent": (
                    f"{overhead / submit * 100.0:.6f}" if submit else "0"
                ),
            })


if __name__ == "__main__":
    main()
