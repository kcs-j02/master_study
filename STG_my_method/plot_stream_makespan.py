#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} input.csv output.png",
            file=sys.stderr,
        )
        return 1

    csv_path = Path(sys.argv[1])
    png_path = Path(sys.argv[2])

    makespans = []
    selected_flags = []
    labels = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            stream_count = int(row["stream_count"])
            sm_counts = row["sm_counts"]

            makespans.append(
                float(row["estimated_makespan"])
            )

            selected_flags.append(
                int(row["selected"]) != 0
            )

            labels.append(
                f"{stream_count}\n{sm_counts}"
            )

    if not makespans:
        print(
            "No stream comparison data found.",
            file=sys.stderr,
        )
        return 1

    # 6候補をそれぞれ別のx座標に置く
    x_positions = list(range(len(makespans)))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        x_positions,
        makespans,
        marker="o",
        linewidth=2,
    )

    for x, y, selected in zip(
        x_positions,
        makespans,
        selected_flags,
    ):
        label = f"{y:.3f}"

        if selected:
            label += "\nSELECTED"

            ax.scatter(
                [x],
                [y],
                s=180,
                marker="*",
                zorder=5,
            )

        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
        )

    ax.set_title(
        "Predicted Makespan for Each Stream / SM Configuration",
        fontsize=15,
    )

    ax.set_xlabel(
        "Stream Count / SM Allocation",
        fontsize=13,
    )

    ax.set_ylabel(
        "Estimated Makespan",
        fontsize=13,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        labels,
        fontsize=10,
    )

    ax.tick_params(
        axis="y",
        labelsize=11,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        png_path,
        dpi=180,
    )

    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())