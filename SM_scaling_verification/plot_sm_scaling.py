#!/usr/bin/env python3

import csv
import subprocess
import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt


def read_measurements(executable: Path) -> dict[str, list[tuple[int, float]]]:
    completed = subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )

    measurements: dict[str, list[tuple[int, float]]] = {
        "LIGHT": [],
        "HEAVY": [],
    }

    for row in csv.reader(StringIO(completed.stdout)):
        if len(row) != 6 or row[0] not in measurements:
            continue

        sm_count = int(row[1])
        if sm_count > 112 or sm_count % 8 != 0:
            continue

        measurements[row[0]].append((sm_count, float(row[2])))

    for kind, values in measurements.items():
        values.sort()
        expected_sm_counts = list(range(8, 113, 8))
        measured_sm_counts = [sm_count for sm_count, _ in values]
        if measured_sm_counts != expected_sm_counts:
            raise RuntimeError(
                f"{kind}: expected SM counts {expected_sm_counts}, "
                f"but got {measured_sm_counts}"
            )

    return measurements


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(
            f"Usage: {sys.argv[0]} benchmark_executable [output.png]",
            file=sys.stderr,
        )
        return 1

    executable = Path(sys.argv[1]).resolve()
    output_path = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) == 3
        else Path(__file__).resolve().parent / "sm_scaling_light_heavy.png"
    )

    measurements = read_measurements(executable)

    fig, ax = plt.subplots(figsize=(12, 7))
    styles = {
        "LIGHT": {"color": "tab:blue", "marker": "o"},
        "HEAVY": {"color": "tab:red", "marker": "s"},
    }

    for kind in ("LIGHT", "HEAVY"):
        values = measurements[kind]
        ax.plot(
            [sm_count for sm_count, _ in values],
            [elapsed_ms for _, elapsed_ms in values],
            linewidth=2.2,
            markersize=6,
            label=f"{kind} task",
            **styles[kind],
        )

    ax.set_title(
        "Measured Processing Time vs Allocated SMs\n"
        "LIGHT and HEAVY Tasks (Median of 21 CUDA-event Measurements)"
    )
    ax.set_xlabel("Allocated SM Count")
    ax.set_ylabel("Median Kernel Time [ms]")
    ax.set_xticks(list(range(8, 113, 8)))
    ax.set_xlim(5, 115)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
