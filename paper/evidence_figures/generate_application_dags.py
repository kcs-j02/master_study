#!/usr/bin/env python3
"""Generate readable DAG figures directly from the KESCO STG definitions."""

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/evidence_figures/application_dags"
STG_ROOT = ROOT / "benchmarks/KESCO"
BENCHMARKS = (
    "vector_square",
    "black_scholes",
    "machine_learning",
    "hits",
    "image_processing",
    "deep_learning",
)


def parse_stg(name):
    path = STG_ROOT / f"BENCHMARK_{name}" / f"{name}.stg"
    lines = path.read_text(encoding="utf-8").splitlines()
    tasks = {}
    comments = {}
    for line in lines:
        match = re.match(
            r"^\s*(\d+)\s+([0-9.]+)\s+(\d+)(?:\s+((?:\d+\s*)*))?\s*(?:#\s*(.*))?$",
            line,
        )
        if not match:
            continue
        task_id, duration, count, predecessors, comment = match.groups()
        predecessor_list = (
            [int(value) for value in predecessors.split()] if predecessors else []
        )
        tasks[int(task_id)] = {
            "duration": float(duration),
            "predecessors": predecessor_list[: int(count)],
        }
        comments[int(task_id)] = comment or f"task {task_id}"
    return tasks, comments


def short_label(comment):
    comment = comment.replace("host reset/barrier", "barrier")
    comment = comment.replace("softmax(r1)", "softmax 1")
    comment = comment.replace("softmax(r2)", "softmax 2")
    comment = comment.replace("conv2d(", "conv ")
    comment = comment.replace("mean_pooling(", "pool ")
    return comment.replace(")", "").replace("(", "")


def positions(name, tasks):
    if name == "hits":
        coordinates = {}
        for task_id in tasks:
            iteration, stage = divmod(task_id, 7)
            if stage == 0:
                coordinates[task_id] = (iteration * 4, 0.75)
            elif stage == 1:
                coordinates[task_id] = (iteration * 4, -0.75)
            elif stage in (2, 4):
                coordinates[task_id] = (iteration * 4 + stage - 1, 0.75)
            elif stage in (3, 5):
                coordinates[task_id] = (iteration * 4 + stage - 1, -0.75)
            else:
                coordinates[task_id] = (iteration * 4 + 3, 0)
        return coordinates
    children = {task_id: [] for task_id in tasks}
    for task_id, task in tasks.items():
        for predecessor in task["predecessors"]:
            children[predecessor].append(task_id)
    level = {}

    def get_level(task_id):
        if task_id in level:
            return level[task_id]
        level[task_id] = (
            0
            if not tasks[task_id]["predecessors"]
            else 1 + max(get_level(item) for item in tasks[task_id]["predecessors"])
        )
        return level[task_id]

    for task_id in tasks:
        get_level(task_id)
    by_level = {}
    for task_id, task_level in level.items():
        by_level.setdefault(task_level, []).append(task_id)
    coordinates = {}
    for task_level, task_ids in by_level.items():
        task_ids.sort()
        count = len(task_ids)
        for index, task_id in enumerate(task_ids):
            coordinates[task_id] = (task_level, -(index - (count - 1) / 2))
    return coordinates


def draw(name):
    tasks, comments = parse_stg(name)
    coordinates = positions(name, tasks)
    durations = [task["duration"] for task in tasks.values()]
    normalizer = Normalize(vmin=min(durations), vmax=max(durations))
    width = max(10, max(x for x, _ in coordinates.values()) * 1.1 + 4)
    height = max(5.5, max(abs(y) for _, y in coordinates.values()) * 1.2 + 4)
    figure, axis = plt.subplots(figsize=(width, height))
    axis.set_title(f"{name.replace('_', ' ').title()} DAG", fontsize=16, pad=18)

    for task_id, task in tasks.items():
        x, y = coordinates[task_id]
        for predecessor in task["predecessors"]:
            px, py = coordinates[predecessor]
            axis.add_patch(
                FancyArrowPatch(
                    (px + 0.22, py),
                    (x - 0.22, y),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.1,
                    color="#59636e",
                    connectionstyle="arc3,rad=0.03",
                    zorder=1,
                )
            )

    for task_id, task in tasks.items():
        x, y = coordinates[task_id]
        color = plt.cm.YlOrRd(normalizer(task["duration"]))
        axis.scatter(
            [x], [y], s=1450 if len(tasks) <= 11 else 850,
            color=color, edgecolor="#263238", linewidth=1.1, zorder=2,
        )
        label = f"{task_id}: {short_label(comments[task_id])}\n{task['duration']:g} us"
        axis.text(x, y, label, ha="center", va="center", fontsize=7 if len(tasks) > 11 else 8,
                  zorder=3)

    axis.set_xlim(-0.7, max(x for x, _ in coordinates.values()) + 0.7)
    ys = [y for _, y in coordinates.values()]
    axis.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    axis.set_xlabel("DAG level (dependency direction left to right)")
    axis.set_yticks([])
    axis.grid(axis="x", alpha=0.16)
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.text(
        0.01, 0.01,
        "Node label: task ID, kernel, representative processing time. "
        "Arrow: dependency. Darker red: longer task.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(OUT / f"{name}_dag.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for benchmark in BENCHMARKS:
        draw(benchmark)


if __name__ == "__main__":
    main()
