#!/usr/bin/env python3

import csv
import re
import shutil
import subprocess
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


# ============================================================
# 基本設定
# ============================================================

# run_method_comparison.py がある場所
#
# /home/kobayashi/main/master_study
#
ROOT = Path(__file__).resolve().parent


# Taskflow
TASKFLOW_ROOT = Path("/home/kobayashi/taskflow")


# ============================================================
# 全手法で共通の入力STG
#
# master_study/
# ├── common_sample.stg
# ├── run_method_comparison.py
# ├── STG/
# ├── STG_existing_method/
# ├── STG_existing_method_GC/
# └── STG_my_method/
# ============================================================

INPUT_FILE = ROOT / "common_sample.stg"


# ============================================================
# NVCC
# ============================================================

NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"


# ============================================================
# 測定回数
#
# 12回実行
# 最初の2回をウォームアップとして除外
# 残り10回の平均
# ============================================================

RUNS = 12
WARMUP_RUNS = 2


# ============================================================
# 比較対象
# ============================================================

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

    cmd = [
        str(x)
        for x in cmd
    ]

    print(
        "$",
        " ".join(cmd)
    )

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if result.returncode != 0:

        print()
        print("===== command output =====")
        print(result.stdout)
        print("==========================")
        print()

        raise RuntimeError(
            f"command failed: {' '.join(cmd)}"
        )

    return result.stdout


# ============================================================
# 入力STG確認
# ============================================================

def check_input_file():

    print("=" * 70)
    print("Input STG")
    print("=" * 70)

    print(
        f"path : {INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"STG file not found: {INPUT_FILE}"
        )

    if not INPUT_FILE.is_file():

        raise RuntimeError(
            f"Input is not a file: {INPUT_FILE}"
        )


    # --------------------------------------------------------
    # STG先頭のタスク数を確認
    # --------------------------------------------------------

    first_nonempty_line = None

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            first_nonempty_line = line

            break


    if first_nonempty_line is None:

        raise RuntimeError(
            f"STG file is empty: {INPUT_FILE}"
        )


    try:

        declared_num_tasks = int(
            first_nonempty_line.split()[0]
        )

    except ValueError:

        raise RuntimeError(
            "Could not parse num_tasks "
            f"from STG: {first_nonempty_line}"
        )


    print(
        f"declared num_tasks : "
        f"{declared_num_tasks}"
    )

    print("=" * 70)
    print()


# ============================================================
# コンパイル
# ============================================================

def build_method(method_cfg):

    method_dir = (
        ROOT /
        method_cfg["dir"]
    )


    source_file = (
        method_dir /
        "main.cu"
    )


    binary_path = (
        method_dir /
        method_cfg["binary"]
    )


    if not method_dir.exists():

        raise FileNotFoundError(
            f"Method directory not found: "
            f"{method_dir}"
        )


    if not source_file.exists():

        raise FileNotFoundError(
            f"main.cu not found: "
            f"{source_file}"
        )


    # --------------------------------------------------------
    # 古い実行ファイル削除
    # --------------------------------------------------------

    if binary_path.exists():

        binary_path.unlink()


    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

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
# gpu_submit_wait_ms
#
# ★ 今回の主評価指標
# ============================================================

def parse_gpu_submit_wait_ms(output):

    match = re.search(
        r"gpu_submit_wait_ms\s*:\s*([0-9.eE+-]+)",
        output,
    )


    if not match:

        print()
        print("===== program output =====")
        print(output)
        print("==========================")
        print()

        raise RuntimeError(
            "gpu_submit_wait_ms could not be parsed"
        )


    return float(
        match.group(1)
    )


# ============================================================
# gpu_kernel_ms
#
# 確認用
# 高速化率には使用しない
# ============================================================

def parse_gpu_kernel_ms(output):

    match = re.search(
        r"gpu_kernel_ms\s*:\s*([0-9.eE+-]+)",
        output,
    )


    if not match:

        return None


    return float(
        match.group(1)
    )


# ============================================================
# num_tasks
#
# 全手法が同じSTGを読んでいるか確認
# ============================================================

def parse_num_tasks(output):

    match = re.search(
        r"num_tasks\s*:\s*([0-9]+)",
        output,
    )


    if not match:

        return None


    return int(
        match.group(1)
    )


# ============================================================
# 1手法を測定
# ============================================================

def run_method(method_cfg):

    method = (
        method_cfg["method"]
    )


    method_dir = (
        ROOT /
        method_cfg["dir"]
    )


    binary_path = (
        method_dir /
        method_cfg["binary"]
    )


    print()
    print("=" * 70)

    print(
        f"Method: {method}"
    )

    print("=" * 70)


    # ========================================================
    # コンパイル
    # ========================================================

    build_method(
        method_cfg
    )


    # ========================================================
    # サンプル保存
    # ========================================================

    submit_wait_samples = []

    kernel_samples = []

    detected_num_tasks = None


    # ========================================================
    # 12回実行
    # ========================================================

    for i in range(RUNS):


        # ----------------------------------------------------
        # INPUT_FILE は絶対パス
        #
        # 全手法が必ず
        #
        # master_study/common_sample.stg
        #
        # を読む
        # ----------------------------------------------------

        output = run_command(

            [

                str(
                    binary_path
                ),

                str(
                    INPUT_FILE
                ),

            ],

            method_dir,
        )


        # ----------------------------------------------------
        # gpu_submit_wait_ms
        # ----------------------------------------------------

        submit_wait_ms = (
            parse_gpu_submit_wait_ms(
                output
            )
        )


        # ----------------------------------------------------
        # gpu_kernel_ms
        # ----------------------------------------------------

        kernel_ms = (
            parse_gpu_kernel_ms(
                output
            )
        )


        # ----------------------------------------------------
        # num_tasks
        # ----------------------------------------------------

        num_tasks = (
            parse_num_tasks(
                output
            )
        )


        if (
            detected_num_tasks is None
            and
            num_tasks is not None
        ):

            detected_num_tasks = (
                num_tasks
            )


        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        submit_wait_samples.append(
            submit_wait_ms
        )


        if kernel_ms is not None:

            kernel_samples.append(
                kernel_ms
            )


        # ----------------------------------------------------
        # 表示
        # ----------------------------------------------------

        print(

            f"[{method}] "

            f"run {i + 1:2d}/{RUNS} "

            f"gpu_submit_wait_ms = "

            f"{submit_wait_ms:.3f} ms",

            end="",
        )


        if kernel_ms is not None:

            print(
                f"  "
                f"gpu_kernel_ms = "
                f"{kernel_ms:.3f} ms"
            )

        else:

            print()


    # ========================================================
    # 最初の2回をウォームアップとして除外
    # ========================================================

    measured_submit_wait_samples = (
        submit_wait_samples[
            WARMUP_RUNS:
        ]
    )


    if not measured_submit_wait_samples:

        raise RuntimeError(
            f"No measured samples: {method}"
        )


    # ========================================================
    # gpu_submit_wait_ms
    # ========================================================

    avg_submit_wait_ms = mean(
        measured_submit_wait_samples
    )


    min_submit_wait_ms = min(
        measured_submit_wait_samples
    )


    max_submit_wait_ms = max(
        measured_submit_wait_samples
    )


    # ========================================================
    # gpu_kernel_ms
    #
    # 参考値
    # ========================================================

    avg_kernel_ms = None


    if len(kernel_samples) == RUNS:

        measured_kernel_samples = (
            kernel_samples[
                WARMUP_RUNS:
            ]
        )


        avg_kernel_ms = mean(
            measured_kernel_samples
        )


    # ========================================================
    # 各手法の結果表示
    # ========================================================

    print()


    print(
        f"[{method}] "
        f"num_tasks = "
        f"{detected_num_tasks}"
    )


    print(
        f"[{method}] "
        f"gpu_submit_wait_ms average = "
        f"{avg_submit_wait_ms:.3f} ms"
    )


    print(
        f"[{method}] "
        f"gpu_submit_wait_ms min     = "
        f"{min_submit_wait_ms:.3f} ms"
    )


    print(
        f"[{method}] "
        f"gpu_submit_wait_ms max     = "
        f"{max_submit_wait_ms:.3f} ms"
    )


    if avg_kernel_ms is not None:

        print(
            f"[{method}] "
            f"gpu_kernel_ms average      = "
            f"{avg_kernel_ms:.3f} ms"
        )


    # ========================================================
    # Result
    # ========================================================

    return {

        "method":
            method,

        "display":
            method_cfg["display"],

        "num_tasks":
            detected_num_tasks,

        "gpu_submit_wait_ms":
            avg_submit_wait_ms,

        "gpu_submit_wait_min_ms":
            min_submit_wait_ms,

        "gpu_submit_wait_max_ms":
            max_submit_wait_ms,

        "gpu_kernel_ms":
            avg_kernel_ms,

    }


# ============================================================
# 全手法のタスク数を確認
# ============================================================

def validate_num_tasks(results):

    values = [

        result["num_tasks"]

        for result in results

        if result["num_tasks"] is not None

    ]


    if not values:

        print(
            "WARNING: "
            "num_tasks could not be parsed."
        )

        return


    first = values[0]


    for value in values:

        if value != first:

            raise RuntimeError(
                "Different num_tasks detected "
                "between methods."
            )


    print()

    print(
        f"[CHECK] all methods used "
        f"{first} tasks"
    )


# ============================================================
# 高速化率
#
#
#              baseline gpu_submit_wait_ms
# speedup = --------------------------------
#               method gpu_submit_wait_ms
#
#
# 1.0 より大きい
# → baselineより高速
#
# 1.0 より小さい
# → baselineより低速
# ============================================================

def calculate_speedup(results):

    baseline = next(

        result

        for result in results

        if result["method"] == "baseline"

    )


    baseline_time = (
        baseline[
            "gpu_submit_wait_ms"
        ]
    )


    if baseline_time <= 0.0:

        raise RuntimeError(
            "Baseline gpu_submit_wait_ms "
            "must be > 0"
        )


    for result in results:


        current_time = (
            result[
                "gpu_submit_wait_ms"
            ]
        )


        if current_time <= 0.0:

            result["speedup"] = 0.0

            result[
                "reduction_percent"
            ] = 0.0

            continue


        # ----------------------------------------------------
        # 高速化率
        # ----------------------------------------------------

        result["speedup"] = (

            baseline_time
            /
            current_time

        )


        # ----------------------------------------------------
        # 実行時間削減率
        #
        # baselineから何%短縮したか
        # ----------------------------------------------------

        result[
            "reduction_percent"
        ] = (

            (
                baseline_time
                -
                current_time
            )

            /

            baseline_time

            *

            100.0
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

        "num_tasks",

        "gpu_submit_wait_ms",

        "gpu_submit_wait_min_ms",

        "gpu_submit_wait_max_ms",

        "gpu_kernel_ms",

        "speedup",

        "reduction_percent",

    ]


    with output_path.open(

        "w",

        newline="",

        encoding="utf-8",

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


                "num_tasks":
                    result["num_tasks"],


                "gpu_submit_wait_ms":
                    f"{result['gpu_submit_wait_ms']:.6f}",


                "gpu_submit_wait_min_ms":
                    f"{result['gpu_submit_wait_min_ms']:.6f}",


                "gpu_submit_wait_max_ms":
                    f"{result['gpu_submit_wait_max_ms']:.6f}",


                "gpu_kernel_ms":
                    (
                        ""
                        if result["gpu_kernel_ms"] is None
                        else
                        f"{result['gpu_kernel_ms']:.6f}"
                    ),


                "speedup":
                    f"{result['speedup']:.6f}",


                "reduction_percent":
                    f"{result['reduction_percent']:.6f}",

            })


    print()

    print(
        f"CSV: {output_path}"
    )


# ============================================================
# gpu_submit_wait_ms 比較グラフ
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
        "GPU Submit Wait Time [ms]"
    )


    plt.title(
        "GPU Submit Wait Time Comparison"
    )


    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )


    # --------------------------------------------------------
    # 棒の上に値を表示
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        times,
    ):

        plt.text(

            bar.get_x()
            +
            bar.get_width() / 2,

            bar.get_height(),

            f"{value:.3f}",

            ha="center",

            va="bottom",

        )


    plt.tight_layout()


    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    png_path = (
        ROOT /
        "gpu_submit_wait_comparison.png"
    )


    plt.savefig(

        png_path,

        dpi=300,

        bbox_inches="tight",

    )


    # --------------------------------------------------------
    # SVG
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Baseline = 1.0
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 棒の上に値
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        speedups,
    ):

        plt.text(

            bar.get_x()
            +
            bar.get_width() / 2,

            bar.get_height(),

            f"{value:.3f}x",

            ha="center",

            va="bottom",

        )


    plt.tight_layout()


    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    png_path = (
        ROOT /
        "speedup_gpu_submit_wait.png"
    )


    plt.savefig(

        png_path,

        dpi=300,

        bbox_inches="tight",

    )


    # --------------------------------------------------------
    # SVG
    # --------------------------------------------------------

    svg_path = (
        ROOT /
        "speedup_gpu_submit_wait.svg"
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

    print("=" * 90)

    print(
        "Final Comparison "
        "(gpu_submit_wait_ms)"
    )

    print("=" * 90)


    print(

        f"{'Method':22s}"

        f"{'Time [ms]':>16s}"

        f"{'Speedup':>14s}"

        f"{'Reduction':>16s}"

    )


    print("-" * 90)


    for result in results:


        print(

            f"{result['method']:22s}"

            f"{result['gpu_submit_wait_ms']:16.3f}"

            f"{result['speedup']:13.3f}x"

            f"{result['reduction_percent']:15.2f}%"

        )


    print("=" * 90)


# ============================================================
# main
# ============================================================

def main():


    # ========================================================
    # STG確認
    # ========================================================

    check_input_file()


    results = []


    # ========================================================
    # 4手法実行
    # ========================================================

    for method_cfg in METHODS:


        result = run_method(
            method_cfg
        )


        results.append(
            result
        )


    # ========================================================
    # 全手法が同じSTGを使用したことを確認
    # ========================================================

    validate_num_tasks(
        results
    )


    # ========================================================
    # gpu_submit_wait_ms を使って高速化率
    # ========================================================

    calculate_speedup(
        results
    )


    # ========================================================
    # 最終結果
    # ========================================================

    print_results(
        results
    )


    # ========================================================
    # CSV
    # ========================================================

    save_csv(
        results
    )


    # ========================================================
    # グラフ
    # ========================================================

    plot_execution_time(
        results
    )


    plot_speedup(
        results
    )


    # ========================================================
    # 最終情報
    # ========================================================

    print()

    print("=" * 70)

    print(
        "All done."
    )

    print("=" * 70)


    print(
        f"Input STG : "
        f"{INPUT_FILE}"
    )


    print(
        "Metric    : "
        "gpu_submit_wait_ms"
    )


    print(
        f"Runs      : "
        f"{RUNS}"
    )


    print(
        f"Warmup    : "
        f"{WARMUP_RUNS}"
    )


    print(
        f"Average   : "
        f"{RUNS - WARMUP_RUNS} runs"
    )


    print()

    print(
        "Speedup = "
        "baseline gpu_submit_wait_ms "
        "/ method gpu_submit_wait_ms"
    )


if __name__ == "__main__":
    main()