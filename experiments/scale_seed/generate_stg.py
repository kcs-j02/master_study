#!/usr/bin/env python3
"""Generate deterministic synthetic STGs for scale and seed experiments."""

import argparse
import random
from pathlib import Path


def task_line(task_id, proc_time, predecessors):
    values = [str(task_id), str(proc_time), str(len(predecessors))]
    values.extend(str(value) for value in predecessors)
    return " ".join(values)


def generate(kind, count, seed):
    rng = random.Random(seed)
    predecessors = [[] for _ in range(count)]
    if kind == "chain":
        for task_id in range(1, count):
            predecessors[task_id] = [task_id - 1]
    elif kind == "fully_parallel":
        for task_id in range(1, count):
            predecessors[task_id] = [0]
    elif kind == "random_dag":
        for task_id in range(1, count):
            candidates = list(range(task_id))
            max_predecessors = min(3, len(candidates))
            predecessor_count = rng.randint(0, max_predecessors)
            predecessors[task_id] = sorted(
                rng.sample(candidates, predecessor_count)
            )
    else:
        raise ValueError(f"unknown graph kind: {kind}")

    lines = [str(count), task_line(0, 0, [])]
    for task_id in range(1, count):
        proc_time = rng.choice((350, 550, 750, 950, 1150, 1350))
        lines.append(task_line(task_id, proc_time, predecessors[task_id]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("chain", "fully_parallel", "random_dag"):
        for count in args.sizes:
            if count < 2:
                raise ValueError("graph size must be at least 2")
            for seed in args.seeds:
                path = args.output_dir / f"{kind}_n{count}_seed{seed}.stg"
                path.write_text(
                    generate(kind, count, seed),
                    encoding="utf-8",
                )


if __name__ == "__main__":
    main()
