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
                f"{stream_count} streams\nFinal SM: {sm_counts}"
            )

    if not makespans:
        print(
            "No stream comparison data found.",
            file=sys.stderr,
        )
        return 1

    # 1～5 Streamの最終makespanをそれぞれ別のx座標に置く。
    x_positions = list(range(len(makespans)))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        x_positions,
        makespans,
        marker="o",
        linewidth=2,
        label="Final makespan after Stage 2",
    )

    for x, y, selected in zip(
        x_positions,
        makespans,
        selected_flags,
    ):
        label = f"Final: {y:.3f}"

        if selected:
            label += "\nMINIMUM / SELECTED"

            ax.scatter(
                [x],
                [y],
                s=180,
                marker="*",
                zorder=5,
            )

        annotation_offset = (12, 16) if selected else (0, 12)
        horizontal_alignment = "left" if selected else "center"

        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=annotation_offset,
            ha=horizontal_alignment,
            fontsize=10,
        )

    ax.set_title(
        "Final Predicted Makespan After Stage 2",
        fontsize=15,
    )

    ax.set_xlabel(
        "Stream Count / Final SM Allocation",
        fontsize=13,
    )

    ax.set_ylabel(
        "Final Estimated Makespan [relative units]",
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

    ax.legend()

    minimum_makespan = min(makespans)
    maximum_makespan = max(makespans)
    makespan_range = maximum_makespan - minimum_makespan
    if makespan_range <= 0.0:
        makespan_range = max(abs(maximum_makespan) * 0.1, 1.0)
    ax.set_ylim(
        minimum_makespan - makespan_range * 0.18,
        maximum_makespan + makespan_range * 0.28,
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
