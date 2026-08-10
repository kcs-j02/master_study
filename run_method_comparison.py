#!/usr/bin/env python3

import csv
import re
import shutil
import subprocess
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================

ROOT = Path(__file__).resolve().parent

TASKFLOW_ROOT = "/home/kobayashi/taskflow"
INPUT_FILE = "sample.stg"

NVCC = shutil.which("nvcc") or "nvcc"

# 合計12回実行
RUNS = 12

# 最初の2回をウォームアップとして除外
WARMUP_RUNS = 2


METHODS = [
    {
        "method": "baseline",
        "display": "Baseline",
        "dir": "STG",
        "binary": "main",
    },
    {
        "method": "existing_method",
        "display": "Existing",
        "dir": "STG_existing_method",
        "binary": "main",
    },
    {
        "method": "existing_method_gc",
        "display": "Existing + GC",
        "dir": "STG_existing_method_GC",
        "binary": "main",
    },
    {
        "method": "proposed",
        "display": "Proposed",
        "dir": "STG_my_method",
        "binary": "main",
    },
]


# ============================================================
# コマンド実行
# ============================================================

def run_command(cmd, cwd):

    print("$", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(result.stdout)

        raise RuntimeError(
            f"command failed: {' '.join(cmd)}"
        )

    return result.stdout


# ============================================================
# コンパイル
# ============================================================

def build_method(method_cfg):

    method_dir = ROOT / method_cfg["dir"]

    cmd = [
        NVCC,
        "-O2",
        "-std=c++20",
        "main.cu",
        f"-I{TASKFLOW_ROOT}",
        "-o",
        method_cfg["binary"],
    ]

    run_command(
        cmd,
        method_dir,
    )


# ============================================================
# gpu_submit_wait_ms を取得
# ============================================================

def parse_gpu_submit_wait_ms(output):

    match = re.search(
        r"gpu_submit_wait_ms\s*:\s*([0-9.eE+-]+)",
        output,
    )

    if not match:

        print(output)

        raise RuntimeError(
            "gpu_submit_wait_ms could not be parsed"
        )

    return float(
        match.group(1)
    )


# ============================================================
# 1手法を測定
# ============================================================

def run_method(method_cfg):

    method = method_cfg["method"]

    method_dir = (
        ROOT /
        method_cfg["dir"]
    )

    binary_path = (
        method_dir /
        method_cfg["binary"]
    )

    print()
    print("=" * 60)
    print(f"Method: {method}")
    print("=" * 60)

    # --------------------------------------------------------
    # コンパイル
    # --------------------------------------------------------

    build_method(
        method_cfg
    )

    samples = []

    # --------------------------------------------------------
    # 12回実行
    # --------------------------------------------------------

    for i in range(RUNS):

        output = run_command(
            [
                str(binary_path),
                INPUT_FILE,
            ],
            method_dir,
        )

        time_ms = (
            parse_gpu_submit_wait_ms(
                output
            )
        )

        samples.append(
            time_ms
        )

        print(
            f"[{method}] "
            f"run {i + 1}/{RUNS} "
            f"gpu_submit_wait_ms = "
            f"{time_ms:.3f} ms"
        )

    # --------------------------------------------------------
    # 最初の2回を捨てる
    # --------------------------------------------------------

    measured_samples = (
        samples[WARMUP_RUNS:]
    )

    if not measured_samples:
        raise RuntimeError(
            f"No measured samples: {method}"
        )

    # --------------------------------------------------------
    # 平均
    # --------------------------------------------------------

    avg_ms = mean(
        measured_samples
    )

    min_ms = min(
        measured_samples
    )

    max_ms = max(
        measured_samples
    )

    print()
    print(
        f"[{method}] "
        f"average = {avg_ms:.3f} ms"
    )

    print(
        f"[{method}] "
        f"min     = {min_ms:.3f} ms"
    )

    print(
        f"[{method}] "
        f"max     = {max_ms:.3f} ms"
    )

    return {
        "method": method,
        "display": method_cfg["display"],
        "gpu_submit_wait_ms": round(
            avg_ms,
            3,
        ),
        "min_ms": round(
            min_ms,
            3,
        ),
        "max_ms": round(
            max_ms,
            3,
        ),
    }


# ============================================================
# 高速化率
# ============================================================

def calculate_speedup(results):

    baseline = next(
        result
        for result in results
        if result["method"] == "baseline"
    )

    baseline_time = (
        baseline["gpu_submit_wait_ms"]
    )

    for result in results:

        current_time = (
            result["gpu_submit_wait_ms"]
        )

        # 高速化率
        #
        #        baseline時間
        # --------------------------------
        #         各手法の時間

        result["speedup"] = round(
            baseline_time / current_time,
            4,
        )


# ============================================================
# CSV保存
# ============================================================

def save_csv(results):

    output_path = (
        ROOT /
        "compare_gpu_submit_wait.csv"
    )

    fieldnames = [
        "method",
        "gpu_submit_wait_ms",
        "speedup",
        "min_ms",
        "max_ms",
    ]

    with output_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                "method":
                    result["method"],

                "gpu_submit_wait_ms":
                    result["gpu_submit_wait_ms"],

                "speedup":
                    result["speedup"],

                "min_ms":
                    result["min_ms"],

                "max_ms":
                    result["max_ms"],
            })

    print()
    print(
        f"CSV: {output_path}"
    )


# ============================================================
# 実行時間グラフ
# ============================================================

def plot_execution_time(results):

    labels = [
        result["display"]
        for result in results
    ]

    times = [
        result["gpu_submit_wait_ms"]
        for result in results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        labels,
        times,
    )

    plt.xlabel(
        "Method"
    )

    plt.ylabel(
        "Execution Time [ms]"
    )

    plt.title(
        "GPU Submit Wait Time"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    # 棒の上に実行時間を表示
    for bar, value in zip(
        bars,
        times,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"{value:.3f}",

            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # PNG
    png_path = (
        ROOT /
        "gpu_submit_wait_comparison.png"
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    # SVG
    svg_path = (
        ROOT /
        "gpu_submit_wait_comparison.svg"
    )

    plt.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Graph: {png_path}"
    )

    print(
        f"Graph: {svg_path}"
    )


# ============================================================
# 高速化率グラフ
# ============================================================

def plot_speedup(results):

    labels = [
        result["display"]
        for result in results
    ]

    speedups = [
        result["speedup"]
        for result in results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        labels,
        speedups,
    )

    # baseline = 1.0
    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Method"
    )

    plt.ylabel(
        "Speedup [x]"
    )

    plt.title(
        "Speedup Based on GPU Submit Wait Time"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    # 棒の上に高速化率を表示
    for bar, value in zip(
        bars,
        speedups,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"{value:.3f}x",

            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # PNG
    png_path = (
        ROOT /
        "speedup_comparison.png"
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    # SVG
    svg_path = (
        ROOT /
        "speedup_comparison.svg"
    )

    plt.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Graph: {png_path}"
    )

    print(
        f"Graph: {svg_path}"
    )


# ============================================================
# 最終結果表示
# ============================================================

def print_results(results):

    print()
    print("=" * 65)
    print("Final Comparison")
    print("=" * 65)

    print(
        f"{'Method':20s}"
        f"{'Time [ms]':>15s}"
        f"{'Speedup':>15s}"
    )

    print("-" * 65)

    for result in results:

        print(
            f"{result['method']:20s}"
            f"{result['gpu_submit_wait_ms']:15.3f}"
            f"{result['speedup']:14.3f}x"
        )

    print("=" * 65)


# ============================================================
# main
# ============================================================

def main():

    results = []

    # --------------------------------------------------------
    # 4手法を測定
    # --------------------------------------------------------

    for method_cfg in METHODS:

        result = run_method(
            method_cfg
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # 高速化率
    # --------------------------------------------------------

    calculate_speedup(
        results
    )

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print_results(
        results
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    save_csv(
        results
    )

    # --------------------------------------------------------
    # グラフ
    # --------------------------------------------------------

    plot_execution_time(
        results
    )

    plot_speedup(
        results
    )

    print()
    print("All done.")


if __name__ == "__main__":
    main()