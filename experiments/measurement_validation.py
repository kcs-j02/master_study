#!/usr/bin/env python3
"""Compare wall-clock submission timing with CUDA-event kernel timing."""

import csv
import re
import sys
from pathlib import Path


VALUE_PATTERN = re.compile(
    r"^(gpu_submit_wait_ms|gpu_kernel_ms):\s+([0-9.eE+-]+)\s+ms$"
)


def parse_log(path):
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = VALUE_PATTERN.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    missing = {"gpu_submit_wait_ms", "gpu_kernel_ms"} - values.keys()
    if missing:
        raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")
    return values


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: measurement_validation.py LOG_ROOT OUTPUT_CSV"
        )

    log_root, output_path = map(Path, sys.argv[1:])
    rows = []
    for path in sorted(log_root.rglob("*.log")):
        values = parse_log(path)
        rows.append({
            "log": str(path),
            "gpu_submit_wait_ms": f"{values['gpu_submit_wait_ms']:.9f}",
            "gpu_kernel_ms": f"{values['gpu_kernel_ms']:.9f}",
            "submit_minus_kernel_ms": (
                f"{values['gpu_submit_wait_ms'] - values['gpu_kernel_ms']:.9f}"
            ),
        })

    if not rows:
        raise SystemExit(f"no log files found below {log_root}")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"measurement validation: {output_path}")


if __name__ == "__main__":
    main()
