#!/usr/bin/env python3
"""Summarize why image processing and deep learning respond differently."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/evidence_figures"
OUT.mkdir(parents=True, exist_ok=True)


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def graph_metrics(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    tasks = {}
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 3 or fields[0].startswith("#"):
            continue
        task_id = int(fields[0])
        proc_time = float(fields[1])
        predecessor_count = int(fields[2])
        tasks[task_id] = (proc_time, [int(value) for value in fields[3:3 + predecessor_count]])
    successors = {task_id: [] for task_id in tasks}
    for task_id, (_, predecessors) in tasks.items():
        for predecessor in predecessors:
            successors[predecessor].append(task_id)
    bottom = {}

    def calculate(task_id):
        if task_id in bottom:
            return bottom[task_id]
        successors_time = [calculate(child) for child in successors[task_id]]
        bottom[task_id] = tasks[task_id][0] + max(successors_time, default=0.0)
        return bottom[task_id]

    for task_id in tasks:
        calculate(task_id)
    levels = {}
    for task_id, (_, predecessors) in tasks.items():
        level = 0
        pending = list(predecessors)
        while pending:
            predecessor = pending.pop()
            level = max(level, levels.get(predecessor, 0) + 1)
        levels[task_id] = level
    widths = {}
    for level in levels.values():
        widths[level] = widths.get(level, 0) + 1
    total_work = sum(value[0] for value in tasks.values())
    return {
        "tasks": len(tasks),
        "edges": sum(len(value[1]) for value in tasks.values()),
        "roots": sum(not value[1] for value in tasks.values()),
        "leaves": sum(not successors[task_id] for task_id in tasks),
        "levels": len(widths),
        "max_level_width": max(widths.values()),
        "total_work": total_work,
        "critical_path": max(bottom.values()),
        "parallelism_bound": total_work / max(bottom.values()),
    }


def write_metrics():
    rows = []
    for benchmark in ("image_processing", "deep_learning"):
        metrics = graph_metrics(
            ROOT / f"benchmarks/KESCO/BENCHMARK_{benchmark}/{benchmark}.stg"
        )
        metrics["benchmark"] = benchmark
        rows.append(metrics)
    fields = [
        "benchmark", "tasks", "edges", "roots", "leaves", "levels",
        "max_level_width", "total_work", "critical_path", "parallelism_bound",
    ]
    output = ROOT / "results/application_stream_sensitivity/graph_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    rows = read(ROOT / "results/application_stream_sensitivity/stream_sensitivity.csv")
    metrics = write_metrics()
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for benchmark, color in (
        ("image_processing", "#1f77b4"),
        ("deep_learning", "#d62728"),
    ):
        selected = [row for row in rows if row["benchmark"] == benchmark]
        x = [int(row["stream_count"]) for row in selected]
        y = [float(row["mean_gpu_kernel_ms"]) for row in selected]
        error = [
            float(row["ci95_gpu_kernel_ms"])
            for row in selected
            if row.get("ci95_gpu_kernel_ms")
        ]
        axis.errorbar(
            x, y, yerr=error if len(error) == len(y) else None,
            marker="o", linewidth=2, capsize=4,
            label=benchmark.replace("_", " ").title(), color=color,
        )
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.set_xlabel("Fixed Stream count")
    axis.set_ylabel("gpu_kernel_ms (ms)")
    axis.set_title("Stream-count sensitivity")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "image_deep_stream_sensitivity.png", dpi=220)
    plt.close(fig)

    summary = OUT / "image_deep_analysis.md"
    image = next(row for row in metrics if row["benchmark"] == "image_processing")
    deep = next(row for row in metrics if row["benchmark"] == "deep_learning")
    summary.write_text(
        "# Image processing and Deep Learning analysis\n\n"
        "## Graph structure\n\n"
        "| Metric | Image processing | Deep Learning |\n"
        "|---|---:|---:|\n"
        f"| Tasks | {image['tasks']} | {deep['tasks']} |\n"
        f"| Edges | {image['edges']} | {deep['edges']} |\n"
        f"| Roots | {image['roots']} | {deep['roots']} |\n"
        f"| Leaves | {image['leaves']} | {deep['leaves']} |\n"
        f"| Levels | {image['levels']} | {deep['levels']} |\n"
        f"| Maximum level width | {image['max_level_width']} | {deep['max_level_width']} |\n"
        f"| Total work | {image['total_work']:.0f} | {deep['total_work']:.0f} |\n"
        f"| Critical path | {image['critical_path']:.0f} | {deep['critical_path']:.0f} |\n"
        f"| Work / critical path | {image['parallelism_bound']:.2f} | {deep['parallelism_bound']:.2f} |\n\n"
        "## Interpretation\n\n"
        "Image processing has more tasks and a wider branching structure. "
        "The sensitivity experiment shows a clear improvement through four Streams, "
        "consistent with useful independent image-processing branches. "
        "Deep Learning has two long convolution branches that merge at concat and "
        "dot-product tasks. Its best fixed Stream count is also four, but the "
        "improvement is smaller and variable; the existing Green Context method "
        "remains faster in the four-method comparison. Therefore, the current "
        "evidence supports a conditional claim: the proposed allocation is useful "
        "when branch-level parallelism and task durations are balanced enough for "
        "the selected SM partition to reduce the critical-path bottleneck, but it "
        "does not dominate for the Deep Learning graph.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
