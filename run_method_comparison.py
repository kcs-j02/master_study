#!/usr/bin/env python3

import re
import shutil
import subprocess
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 基本設定
# ============================================================

# このスクリプトが存在するディレクトリ
#
# /home/kobayashi/main/master_study
#
ROOT = Path(__file__).resolve().parent


# Taskflow
TASKFLOW_ROOT = Path(
    "/home/kobayashi/taskflow"
)


# NVCC
NVCC = (
    shutil.which("nvcc")
    or "/usr/local/cuda/bin/nvcc"
)


# ============================================================
# 評価対象STG
#
# すべて run_method_comparison.py と同じ階層
# ============================================================

INPUT_FILES = [

    ROOT / "common_sample.stg",

    ROOT / "sample_fully_parallel.stg",

    ROOT / "sample_Multiple_long_branches.stg",

    ROOT / "sample_random.stg",

    ROOT / "sample_trial.stg",

]


# ============================================================
# 測定回数
#
# 合計12回
# 最初の2回をウォームアップとして除外
# 残り10回の平均を評価値とする
# ============================================================

RUNS = 12

WARMUP_RUNS = 2


# ============================================================
# 評価対象手法
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

def run_command(
    cmd,
    cwd,
):

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

        print(
            "===== command output ====="
        )

        print(
            result.stdout
        )

        print(
            "=========================="
        )

        print()


        raise RuntimeError(
            f"command failed: {' '.join(cmd)}"
        )


    return result.stdout


# ============================================================
# STG入力ファイル確認
# ============================================================

def check_input_files():

    print()

    print(
        "=" * 80
    )

    print(
        "Input STG Files"
    )

    print(
        "=" * 80
    )


    for input_file in INPUT_FILES:


        # ----------------------------------------------------
        # ファイル存在確認
        # ----------------------------------------------------

        if not input_file.exists():

            raise FileNotFoundError(
                f"STG file not found: "
                f"{input_file}"
            )


        if not input_file.is_file():

            raise RuntimeError(
                f"Not a file: "
                f"{input_file}"
            )


        # ----------------------------------------------------
        # 先頭行からタスク数確認
        # ----------------------------------------------------

        first_line = None


        with input_file.open(
            "r",
            encoding="utf-8",
        ) as f:


            for line in f:

                line = line.strip()


                if not line:

                    continue


                if line.startswith("#"):

                    continue


                first_line = line

                break


        if first_line is None:

            raise RuntimeError(
                f"STG file is empty: "
                f"{input_file}"
            )


        print(

            f"{input_file.name:40s}"

            f"num_tasks = "

            f"{first_line}"

        )


    print(
        "=" * 80
    )

    print()


# ============================================================
# 全手法をコンパイル
#
# STGごとにコンパイルすると無駄なので、
# プログラム開始時に4手法を1回ずつだけコンパイルする
# ============================================================

def build_all_methods():

    print()

    print(
        "=" * 80
    )

    print(
        "Build Methods"
    )

    print(
        "=" * 80
    )


    for method_cfg in METHODS:


        method_dir = (
            ROOT
            /
            method_cfg["dir"]
        )


        source_file = (
            method_dir
            /
            "main.cu"
        )


        binary_path = (
            method_dir
            /
            method_cfg["binary"]
        )


        # ----------------------------------------------------
        # ディレクトリ確認
        # ----------------------------------------------------

        if not method_dir.exists():

            raise FileNotFoundError(
                f"Method directory "
                f"not found: "
                f"{method_dir}"
            )


        # ----------------------------------------------------
        # main.cu確認
        # ----------------------------------------------------

        if not source_file.exists():

            raise FileNotFoundError(
                f"main.cu not found: "
                f"{source_file}"
            )


        # ----------------------------------------------------
        # 古いバイナリを削除
        # ----------------------------------------------------

        if binary_path.exists():

            binary_path.unlink()


        print()

        print(
            f"[BUILD] "
            f"{method_cfg['method']}"
        )


        # ----------------------------------------------------
        # Compile
        # ----------------------------------------------------

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


    print()

    print(
        "Build completed."
    )

    print(
        "=" * 80
    )


# ============================================================
# gpu_submit_wait_ms を取得
#
# ★ 主評価指標
#
# CPU側でGPU処理を開始してから
# GPU上の全処理が完了するまでの時間
# ============================================================

def parse_gpu_submit_wait_ms(
    output,
):


    match = re.search(

        r"gpu_submit_wait_ms\s*:\s*"
        r"([0-9.eE+-]+)",

        output,

    )


    if not match:

        print()

        print(
            "===== program output ====="
        )

        print(
            output
        )

        print(
            "=========================="
        )

        print()


        raise RuntimeError(
            "gpu_submit_wait_ms "
            "could not be parsed"
        )


    return float(
        match.group(1)
    )


# ============================================================
# gpu_kernel_ms を取得
#
# 参考値
# 高速化率には使用しない
# ============================================================

def parse_gpu_kernel_ms(
    output,
):


    match = re.search(

        r"gpu_kernel_ms\s*:\s*"
        r"([0-9.eE+-]+)",

        output,

    )


    if not match:

        return None


    return float(
        match.group(1)
    )


# ============================================================
# num_tasks を取得
#
# 4手法が同じSTGを読み込んでいるか確認するため
# ============================================================

def parse_num_tasks(
    output,
):


    match = re.search(

        r"num_tasks\s*:\s*"
        r"([0-9]+)",

        output,

    )


    if not match:

        return None


    return int(
        match.group(1)
    )


# ============================================================
# 1手法 × 1STG を測定
# ============================================================

def run_method_for_stg(
    method_cfg,
    input_file,
):


    method = (
        method_cfg["method"]
    )


    method_dir = (
        ROOT
        /
        method_cfg["dir"]
    )


    binary_path = (
        method_dir
        /
        method_cfg["binary"]
    )


    print()

    print(
        "-" * 80
    )

    print(
        f"STG    : "
        f"{input_file.name}"
    )

    print(
        f"Method : "
        f"{method}"
    )

    print(
        "-" * 80
    )


    # ========================================================
    # 測定値保存
    # ========================================================

    submit_wait_samples = []

    kernel_samples = []

    detected_num_tasks = None


    # ========================================================
    # RUNS回実行
    # ========================================================

    for i in range(RUNS):


        # ----------------------------------------------------
        # 絶対パスでSTGを渡す
        #
        # これにより各methodディレクトリに
        # sample.stg等が存在しても誤って読まない
        # ----------------------------------------------------

        output = run_command(

            [

                str(
                    binary_path
                ),

                str(
                    input_file
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
        # 各run結果表示
        # ----------------------------------------------------

        print(

            f"[{input_file.stem}] "

            f"[{method}] "

            f"run "
            f"{i + 1:2d}/{RUNS} "

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
    # 最初のWARMUP_RUNS回を除外
    # ========================================================

    measured_submit_samples = (

        submit_wait_samples[
            WARMUP_RUNS:
        ]

    )


    if not measured_submit_samples:

        raise RuntimeError(
            f"No measured samples: "
            f"{method}"
        )


    # ========================================================
    # gpu_submit_wait_ms 統計
    # ========================================================

    avg_submit_wait_ms = mean(
        measured_submit_samples
    )


    min_submit_wait_ms = min(
        measured_submit_samples
    )


    max_submit_wait_ms = max(
        measured_submit_samples
    )


    # ========================================================
    # gpu_kernel_ms 平均
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
    # 結果表示
    # ========================================================

    print()


    print(

        f"[RESULT] "

        f"{input_file.stem} "

        f"/ "

        f"{method}"

    )


    print(

        f"  num_tasks                  = "

        f"{detected_num_tasks}"

    )


    print(

        f"  gpu_submit_wait_ms average = "

        f"{avg_submit_wait_ms:.3f} ms"

    )


    print(

        f"  gpu_submit_wait_ms min     = "

        f"{min_submit_wait_ms:.3f} ms"

    )


    print(

        f"  gpu_submit_wait_ms max     = "

        f"{max_submit_wait_ms:.3f} ms"

    )


    if avg_kernel_ms is not None:

        print(

            f"  gpu_kernel_ms average      = "

            f"{avg_kernel_ms:.3f} ms"

        )


    # ========================================================
    # 結果返却
    # ========================================================

    return {

        "stg":
            input_file.stem,

        "stg_file":
            input_file.name,

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
# 同一STGについて
# 全手法のnum_tasksが一致しているか確認
# ============================================================

def validate_num_tasks(
    results,
):


    values = [

        result["num_tasks"]

        for result in results

        if result["num_tasks"]
        is not None

    ]


    if not values:

        print(
            "[WARNING] "
            "num_tasks could not be parsed"
        )

        return


    first = values[0]


    for value in values:

        if value != first:

            raise RuntimeError(
                "Different num_tasks "
                "detected between methods"
            )


    print()

    print(

        f"[CHECK] "

        f"all methods used "

        f"{first} tasks"

    )


# ============================================================
# 高速化率計算
#
#
#                Baseline gpu_submit_wait_ms
# Speedup = -----------------------------------
#                 Method gpu_submit_wait_ms
#
#
# > 1.0
#   Baselineより高速
#
# = 1.0
#   Baselineと同じ
#
# < 1.0
#   Baselineより低速
# ============================================================

def calculate_speedup(
    results,
):


    baseline = next(

        result

        for result in results

        if result["method"]
        ==
        "baseline"

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

            result[
                "speedup"
            ] = 0.0

            result[
                "reduction_percent"
            ] = 0.0

            continue


        # ----------------------------------------------------
        # Speedup
        # ----------------------------------------------------

        result[
            "speedup"
        ] = (

            baseline_time

            /

            current_time

        )


        # ----------------------------------------------------
        # 実行時間削減率
        #
        # Baselineより何%短縮したか
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
# 1STGの実行時間グラフ
#
# PNGのみ出力
# ============================================================

def plot_execution_time(
    stg_name,
    results,
):


    labels = [

        result["display"]

        for result in results

    ]


    times = [

        result[
            "gpu_submit_wait_ms"
        ]

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
        f"Execution Time - {stg_name}"
    )


    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )


    # --------------------------------------------------------
    # 棒グラフ上に実行時間
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

            f"{value:.2f}",

            ha="center",

            va="bottom",

        )


    plt.tight_layout()


    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    output_path = (

        ROOT

        /

        f"{stg_name}_gpu_submit_wait.png"

    )


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight",

    )


    plt.close()


    print(

        f"Graph: "
        f"{output_path}"

    )


# ============================================================
# 1STGのSpeedupグラフ
#
# PNGのみ出力
# ============================================================

def plot_speedup(
    stg_name,
    results,
):


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
        f"Speedup - {stg_name}"
    )


    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )


    # --------------------------------------------------------
    # 棒グラフ上にSpeedup
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

    output_path = (

        ROOT

        /

        f"{stg_name}_speedup.png"

    )


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight",

    )


    plt.close()


    print(

        f"Graph: "
        f"{output_path}"

    )


# ============================================================
# 1STGの結果をコンソール表示
# ============================================================

def print_stg_results(
    stg_name,
    results,
):


    print()

    print(
        "=" * 90
    )


    print(
        f"Final Comparison: "
        f"{stg_name}"
    )


    print(
        "=" * 90
    )


    print(

        f"{'Method':22s}"

        f"{'Time [ms]':>16s}"

        f"{'Speedup':>14s}"

        f"{'Reduction':>16s}"

    )


    print(
        "-" * 90
    )


    for result in results:


        print(

            f"{result['method']:22s}"

            f"{result['gpu_submit_wait_ms']:16.3f}"

            f"{result['speedup']:13.3f}x"

            f"{result['reduction_percent']:15.2f}%"

        )


    print(
        "=" * 90
    )


# ============================================================
# 全STGのSpeedupを1枚にまとめる
#
# PNGのみ
# ============================================================

def plot_all_speedups(
    all_stg_results,
):


    stg_names = list(
        all_stg_results.keys()
    )


    x = np.arange(
        len(stg_names)
    )


    # 4手法なので幅0.18
    width = 0.18


    plt.figure(
        figsize=(13, 6)
    )


    # ========================================================
    # 各手法
    # ========================================================

    for method_index, method_cfg in enumerate(
        METHODS
    ):


        values = []


        # ----------------------------------------------------
        # 各STGにおけるSpeedupを取得
        # ----------------------------------------------------

        for stg_name in stg_names:


            result = next(

                result

                for result
                in all_stg_results[
                    stg_name
                ]

                if result["method"]
                ==
                method_cfg["method"]

            )


            values.append(
                result["speedup"]
            )


        # ----------------------------------------------------
        # 棒の位置
        # ----------------------------------------------------

        offset = (

            method_index

            -

            (
                len(METHODS) - 1
            )
            / 2

        ) * width


        bars = plt.bar(

            x + offset,

            values,

            width,

            label=
                method_cfg[
                    "display"
                ],

        )


        # ----------------------------------------------------
        # 数値表示
        # ----------------------------------------------------

        for bar, value in zip(
            bars,
            values,
        ):


            plt.text(

                bar.get_x()
                +
                bar.get_width() / 2,

                bar.get_height(),

                f"{value:.2f}",

                ha="center",

                va="bottom",

                fontsize=8,

            )


    # ========================================================
    # Baseline = 1
    # ========================================================

    plt.axhline(

        y=1.0,

        linestyle="--",

        linewidth=1,

    )


    plt.xlabel(
        "STG"
    )


    plt.ylabel(
        "Speedup [x]"
    )


    plt.title(
        "Speedup Comparison Across STGs"
    )


    plt.xticks(

        x,

        stg_names,

        rotation=20,

        ha="right",

    )


    plt.legend()


    plt.grid(

        axis="y",

        linestyle="--",

        alpha=0.4,

    )


    plt.tight_layout()


    # ========================================================
    # PNG
    # ========================================================

    output_path = (

        ROOT

        /

        "all_stg_speedup_comparison.png"

    )


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight",

    )


    plt.close()


    print()

    print(

        f"Graph: "
        f"{output_path}"

    )


# ============================================================
# 全STGの実行時間を1枚にまとめる
#
# gpu_submit_wait_ms
#
# PNGのみ
# ============================================================

def plot_all_execution_times(
    all_stg_results,
):


    stg_names = list(
        all_stg_results.keys()
    )


    x = np.arange(
        len(stg_names)
    )


    width = 0.18


    plt.figure(
        figsize=(13, 6)
    )


    # ========================================================
    # 各手法
    # ========================================================

    for method_index, method_cfg in enumerate(
        METHODS
    ):


        values = []


        for stg_name in stg_names:


            result = next(

                result

                for result
                in all_stg_results[
                    stg_name
                ]

                if result["method"]
                ==
                method_cfg["method"]

            )


            values.append(

                result[
                    "gpu_submit_wait_ms"
                ]

            )


        offset = (

            method_index

            -

            (
                len(METHODS) - 1
            )
            / 2

        ) * width


        bars = plt.bar(

            x + offset,

            values,

            width,

            label=
                method_cfg[
                    "display"
                ],

        )


        # ----------------------------------------------------
        # 数値表示
        # ----------------------------------------------------

        for bar, value in zip(
            bars,
            values,
        ):


            plt.text(

                bar.get_x()
                +
                bar.get_width() / 2,

                bar.get_height(),

                f"{value:.1f}",

                ha="center",

                va="bottom",

                fontsize=8,

            )


    plt.xlabel(
        "STG"
    )


    plt.ylabel(
        "GPU Submit Wait Time [ms]"
    )


    plt.title(
        "Execution Time Comparison Across STGs"
    )


    plt.xticks(

        x,

        stg_names,

        rotation=20,

        ha="right",

    )


    plt.legend()


    plt.grid(

        axis="y",

        linestyle="--",

        alpha=0.4,

    )


    plt.tight_layout()


    # ========================================================
    # PNG
    # ========================================================

    output_path = (

        ROOT

        /

        "all_stg_execution_time_comparison.png"

    )


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight",

    )


    plt.close()


    print(

        f"Graph: "
        f"{output_path}"

    )


# ============================================================
# 全STGの最終結果をまとめて表示
# ============================================================

def print_all_results(
    all_stg_results,
):


    print()

    print()

    print(
        "#" * 95
    )

    print(
        "# ALL STG FINAL RESULTS"
    )

    print(
        "#" * 95
    )


    for stg_name, results in (
        all_stg_results.items()
    ):


        print()

        print(
            f"[{stg_name}]"
        )


        for result in results:


            print(

                f"  "

                f"{result['display']:15s}"

                f" "

                f"{result['gpu_submit_wait_ms']:10.3f} ms"

                f"   "

                f"{result['speedup']:7.3f}x"

                f"   "

                f"{result['reduction_percent']:7.2f}%"

            )


# ============================================================
# main
# ============================================================

def main():


    # ========================================================
    # STG確認
    # ========================================================

    check_input_files()


    # ========================================================
    # 4手法をコンパイル
    #
    # 最初の1回のみ
    # ========================================================

    build_all_methods()


    # ========================================================
    # 全STGの結果
    # ========================================================

    all_stg_results = {}


    # ========================================================
    # STGごとに評価
    # ========================================================

    for input_file in INPUT_FILES:


        stg_name = (
            input_file.stem
        )


        print()

        print()

        print(
            "#" * 90
        )


        print(
            f"# STG: "
            f"{stg_name}"
        )


        print(
            "#" * 90
        )


        results = []


        # ====================================================
        # 4手法
        # ====================================================

        for method_cfg in METHODS:


            result = (
                run_method_for_stg(

                    method_cfg,

                    input_file,

                )
            )


            results.append(
                result
            )


        # ====================================================
        # 全手法が同じタスク数を使用したか確認
        # ====================================================

        validate_num_tasks(
            results
        )


        # ====================================================
        # Speedup計算
        # ====================================================

        calculate_speedup(
            results
        )


        # ====================================================
        # コンソール結果
        # ====================================================

        print_stg_results(

            stg_name,

            results,

        )


        # ====================================================
        # 各STGの実行時間PNG
        # ====================================================

        plot_execution_time(

            stg_name,

            results,

        )


        # ====================================================
        # 各STGのSpeedup PNG
        # ====================================================

        plot_speedup(

            stg_name,

            results,

        )


        # ====================================================
        # 全体用
        # ====================================================

        all_stg_results[
            stg_name
        ] = results


    # ========================================================
    # 全STG結果表示
    # ========================================================

    print_all_results(
        all_stg_results
    )


    # ========================================================
    # 全STGまとめSpeedupグラフ
    # ========================================================

    plot_all_speedups(
        all_stg_results
    )


    # ========================================================
    # 全STGまとめ実行時間グラフ
    # ========================================================

    plot_all_execution_times(
        all_stg_results
    )


    # ========================================================
    # 終了
    # ========================================================

    print()

    print(
        "=" * 90
    )


    print(
        "All STG comparisons completed."
    )


    print(
        "=" * 90
    )


    print(
        "Metric  : "
        "gpu_submit_wait_ms"
    )


    print(

        f"Runs    : "
        f"{RUNS}"

    )


    print(

        f"Warmup  : "
        f"{WARMUP_RUNS}"

    )


    print(

        f"Average : "
        f"{RUNS - WARMUP_RUNS} runs"

    )


    print()


    print(
        "Speedup = "
        "Baseline gpu_submit_wait_ms "
        "/ "
        "Method gpu_submit_wait_ms"
    )


    print()


    print(
        "Generated files: PNG only"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()