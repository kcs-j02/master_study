#!/usr/bin/env python3
"""Plot the measured representative kernel times used as STG task weights."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/evidence_figures"

DATA = {
    "Image Processing": [
        ("0 gaussian small", 190), ("1 gaussian large", 5),
        ("2 gaussian unsharpen", 3), ("3 sobel small", 3),
        ("4 sobel large", 3), ("5 maximum", 10), ("6 minimum", 7),
        ("7 extend", 8), ("8 unsharpen", 9), ("9 combine image2", 7),
        ("10 combine image3", 9),
    ],
    "Deep Learning": [
        ("0 conv layer1 x", 621), ("1 conv layer1 y", 14),
        ("2 mean pool x", 22), ("3 mean pool y", 22),
        ("4 conv layer2 x", 52), ("5 conv layer2 y", 52),
        ("6 concat", 2), ("7 dot product", 1),
    ],
}


def main():
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for axis, (title, values) in zip(axes, DATA.items()):
        labels = [label for label, _ in values]
        times = [time for _, time in values]
        bars = axis.barh(range(len(values)), times, color="#4472c4")
        axis.set_yticks(range(len(values)), labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("Representative measured kernel time (us)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        for bar, time in zip(bars, times):
            axis.text(time + max(times) * 0.015, bar.get_y() + bar.get_height() / 2,
                      f"{time} us", va="center", fontsize=8)
    figure.suptitle("Kernel-level weights used by the STG scheduler", fontsize=15)
    figure.savefig(OUT / "kernel_measurement_mapping.png", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
