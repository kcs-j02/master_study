#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} input.csv output.png actual_makespan_ms",
            file=sys.stderr,
        )
        return 1

    csv_path = Path(sys.argv[1])
    png_path = Path(sys.argv[2])
    actual_makespan_ms = float(sys.argv[3])

    labels = []
    before_processing_times = []
    first_stage_processing_times = []
    second_stage_processing_times = []
    before_sm_counts = []
    first_stage_sm_counts = []
    second_stage_sm_counts = []
    stream_counts = []
    selected_flags = []
    final_makespans = []

    initial_sm_by_stream_count = {
        1: 114,
        2: 56,
        3: 32,
        4: 24,
        5: 16,
    }

    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            stream_count = int(row["stream_count"])
            labels.append(
                f"{stream_count} streams\nStream {row['stream_id']}"
            )
            before_processing_times.append(
                float(row["before_redistribution_processing_time"])
            )
            first_stage_processing_times.append(
                float(row["first_stage_processing_time"])
            )
            second_stage_processing_times.append(
                float(row["relative_load_after_sm_partition"])
            )
            before_sm_counts.append(
                initial_sm_by_stream_count[stream_count]
            )
            first_stage_sm_counts.append(int(row["initial_pool_sm"]))
            second_stage_sm_counts.append(int(row["sm_count"]))
            stream_counts.append(stream_count)
            selected_flags.append(int(row["selected"]) != 0)
            final_makespans.append(float(row["final_makespan"]))

    if not labels:
        print("No stream processing-time data found.", file=sys.stderr)
        return 1

    positions = list(range(len(labels)))
    fig, (ax_prediction, ax_actual) = plt.subplots(
        2,
        1,
        figsize=(max(16, len(labels) * 1.5), 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    width = 0.25
    bars_before = ax_prediction.bar(
        [position - width for position in positions],
        before_processing_times,
        width,
        color="tab:blue",
        label="Before redistribution: initial equal SM allocation",
    )
    bars_first = ax_prediction.bar(
        positions,
        first_stage_processing_times,
        width,
        color="tab:purple",
        label="Stage 1: initial-pool optimization",
    )
    bars_second = ax_prediction.bar(
        [position + width for position in positions],
        second_stage_processing_times,
        width,
        color="tab:orange",
        label="Stage 2: +2 to stream 0, then remainder optimization",
    )

    ax_prediction.bar_label(
        bars_before,
        labels=[
            f"SM={sm_count}"
            for sm_count in before_sm_counts
        ],
        padding=3,
        fontsize=6,
    )
    ax_prediction.bar_label(
        bars_first,
        labels=[
            f"{value:.1f}\n×{(value / before if before else 0.0):.2f}"
            f"\nSM={sm_count}"
            for value, before, sm_count in zip(
                first_stage_processing_times,
                before_processing_times,
                first_stage_sm_counts,
            )
        ],
        padding=3,
        fontsize=6,
    )
    ax_prediction.bar_label(
        bars_second,
        labels=[
            f"{value:.1f}\n×{(value / before if before else 0.0):.2f}"
            f"\nSM={sm_count}"
            for value, before, sm_count in zip(
                second_stage_processing_times,
                before_processing_times,
                second_stage_sm_counts,
            )
        ],
        padding=3,
        fontsize=6,
    )
    ax_prediction.set_title(
        "Processing Time Before / After Each SM Allocation Stage"
    )
    ax_prediction.set_xlabel(
        "Stream Count / Stream"
    )
    ax_prediction.set_ylabel("Estimated Processing Time [relative units]")
    ax_prediction.set_xticks(positions, labels)
    maximum_value = max(
        before_processing_times
        + first_stage_processing_times
        + second_stage_processing_times
        + final_makespans
    )
    ax_prediction.set_ylim(0, maximum_value * 1.42)
    ax_prediction.grid(axis="y", alpha=0.3)
    ax_prediction.legend(loc="upper right")

    for position, selected in zip(positions, selected_flags):
        if selected:
            ax_prediction.axvspan(
                position - 0.48,
                position + 0.48,
                color="gold",
                alpha=0.12,
            )

    # 各StreamのSM補正後処理時間合計の最大値を表示する。
    for stream_count in sorted(set(stream_counts)):
        indices = [
            index
            for index, value in enumerate(stream_counts)
            if value == stream_count
        ]
        makespan = final_makespans[indices[0]]
        group_center = sum(indices) / len(indices)
        ax_prediction.hlines(
            makespan,
            indices[0] - 0.45,
            indices[-1] + 0.45,
            colors="darkred",
            linestyles="--",
            linewidth=1.5,
            label=(
                "Final makespan (maximum stream processing time)"
                if stream_count == min(stream_counts)
                else None
            ),
        )
        ax_prediction.annotate(
            f"Final makespan\n{makespan:.1f}",
            (group_center, makespan),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
            color="darkred",
        )

    ax_prediction.legend(loc="upper right")

    actual_bar = ax_actual.barh(
        ["Measured makespan"],
        [actual_makespan_ms],
        color="tab:green",
    )
    ax_actual.bar_label(
        actual_bar,
        labels=[f"{actual_makespan_ms:.3f} ms"],
        padding=4,
        fontsize=10,
    )
    ax_actual.set_title(
        "Actual Makespan (GPU Submit + Wait)"
    )
    ax_actual.set_xlabel("Actual Makespan [ms]")
    ax_actual.set_xlim(0, actual_makespan_ms * 1.22)
    ax_actual.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
