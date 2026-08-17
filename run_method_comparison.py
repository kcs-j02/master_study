#!/usr/bin/env python3

import re
import shutil
import sqlite3
import subprocess
import os
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 基本設定
# ============================================================

ROOT = Path(__file__).resolve().parent

TASKFLOW_ROOT = Path("/home/kobayashi/taskflow")

NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
NSYS = shutil.which("nsys")

# Nsight Systems の一時ファイル置き場
NSYS_TMP_ROOT = Path.home() / "tmp" / "nsys"
NSYS_TMP_ROOT.mkdir(parents=True, exist_ok=True)

# Nsight Systems レポート保存先
NSYS_RESULT_DIR = ROOT / "nsys_reports"
NSYS_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# GPU Metrics
GPU_METRICS_DEVICE = "0"
GPU_METRICS_FREQUENCY = 10000  # 10 kHz


# ============================================================
# 評価対象STG
# ============================================================

INPUT_FILES = [
    ROOT / "sample_mixed_chain_parallel.stg",
    ROOT / "sample_fully_parallel.stg",
    ROOT / "sample_Multiple_long_branches.stg",
    ROOT / "sample_random.stg",
    ROOT / "sample_trial.stg",
]


STG_DISPLAY_NAMES = {
    "sample_mixed_chain_parallel": "Mixed: Chain+Parallel ",
    "sample_fully_parallel": "Fully Parallel",
    "sample_Multiple_long_branches": "Multiple Long Branches",
    "sample_random": "Random",
    "sample_trial": "Trial",
}


# ============================================================
# 実行時間評価
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
# 共通コマンド実行
# ============================================================

def run_command(cmd, cwd, env=None):
    cmd = [str(x) for x in cmd]

    print("$", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=env,
    )

    if result.returncode != 0:
        print()
        print("===== command output =====")
        print(result.stdout)
        print("==========================")
        print()
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}"
        )

    return result.stdout


# ============================================================
# 入力確認
# ============================================================

def check_input_files():
    print()
    print("=" * 80)
    print("Input STG Files")
    print("=" * 80)

    for input_file in INPUT_FILES:
        if not input_file.exists():
            raise FileNotFoundError(f"STG file not found: {input_file}")

        if not input_file.is_file():
            raise RuntimeError(f"Not a file: {input_file}")

        first_line = None

        with input_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                first_line = line
                break

        if first_line is None:
            raise RuntimeError(f"STG file is empty: {input_file}")

        print(
            f"{input_file.name:40s}"
            f"num_tasks = {first_line}"
        )

    print("=" * 80)
    print()


# ============================================================
# 全手法を1回ずつコンパイル
# ============================================================

def build_all_methods():
    print()
    print("=" * 80)
    print("Build Methods")
    print("=" * 80)

    for method_cfg in METHODS:
        method_dir = ROOT / method_cfg["dir"]
        source_file = method_dir / "main.cu"
        binary_path = method_dir / method_cfg["binary"]

        if not method_dir.exists():
            raise FileNotFoundError(
                f"Method directory not found: {method_dir}"
            )

        if not source_file.exists():
            raise FileNotFoundError(
                f"main.cu not found: {source_file}"
            )

        if binary_path.exists():
            binary_path.unlink()

        print()
        print(f"[BUILD] {method_cfg['method']}")

        cmd = [
            NVCC,
            "-O2",
            "-std=c++20",
            "main.cu",
            f"-I{TASKFLOW_ROOT}",
            "-o",
            method_cfg["binary"],
        ]

        run_command(cmd, method_dir)

    print()
    print("Build completed.")
    print("=" * 80)


# ============================================================
# 通常実行の出力パース
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

    return float(match.group(1))


def parse_gpu_kernel_ms(output):
    match = re.search(
        r"gpu_kernel_ms\s*:\s*([0-9.eE+-]+)",
        output,
    )

    if not match:
        return None

    return float(match.group(1))


def parse_num_tasks(output):
    match = re.search(
        r"num_tasks\s*:\s*([0-9]+)",
        output,
    )

    if not match:
        return None

    return int(match.group(1))


# ============================================================
# PHASE 1:
# 1手法 × 1STG の実行時間測定
# Nsight Systems は使わない
# ============================================================

def run_method_for_stg(method_cfg, input_file):
    method = method_cfg["method"]
    method_dir = ROOT / method_cfg["dir"]
    binary_path = method_dir / method_cfg["binary"]

    print()
    print("-" * 80)
    print(f"STG    : {input_file.name}")
    print(f"Method : {method}")
    print("-" * 80)

    submit_wait_samples = []
    kernel_samples = []
    detected_num_tasks = None

    for i in range(RUNS):
        output = run_command(
            [
                binary_path,
                input_file,
            ],
            method_dir,
        )

        submit_wait_ms = parse_gpu_submit_wait_ms(output)
        kernel_ms = parse_gpu_kernel_ms(output)
        num_tasks = parse_num_tasks(output)

        submit_wait_samples.append(submit_wait_ms)

        if kernel_ms is not None:
            kernel_samples.append(kernel_ms)

        if detected_num_tasks is None and num_tasks is not None:
            detected_num_tasks = num_tasks

        print(
            f"[{input_file.stem}] "
            f"[{method}] "
            f"run {i + 1:2d}/{RUNS} "
            f"gpu_submit_wait_ms = {submit_wait_ms:.3f} ms",
            end="",
        )

        if kernel_ms is not None:
            print(
                f"  gpu_kernel_ms = {kernel_ms:.3f} ms"
            )
        else:
            print()

    measured_submit_samples = submit_wait_samples[WARMUP_RUNS:]

    if not measured_submit_samples:
        raise RuntimeError(
            f"No measured samples: {method}"
        )

    avg_submit_wait_ms = mean(measured_submit_samples)
    min_submit_wait_ms = min(measured_submit_samples)
    max_submit_wait_ms = max(measured_submit_samples)

    avg_kernel_ms = None

    if len(kernel_samples) == RUNS:
        measured_kernel_samples = kernel_samples[WARMUP_RUNS:]
        avg_kernel_ms = mean(measured_kernel_samples)

    print()
    print(f"[RESULT] {input_file.stem} / {method}")
    print(f"  num_tasks                  = {detected_num_tasks}")
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

    return {
        "stg": input_file.stem,
        "stg_file": input_file.name,
        "method": method,
        "display": method_cfg["display"],
        "num_tasks": detected_num_tasks,
        "gpu_submit_wait_ms": avg_submit_wait_ms,
        "gpu_submit_wait_min_ms": min_submit_wait_ms,
        "gpu_submit_wait_max_ms": max_submit_wait_ms,
        "gpu_kernel_ms": avg_kernel_ms,
    }


# ============================================================
# 同一STGでタスク数が一致するか確認
# ============================================================

def validate_num_tasks(results):
    values = [
        result["num_tasks"]
        for result in results
        if result["num_tasks"] is not None
    ]

    if not values:
        print("[WARNING] num_tasks could not be parsed")
        return

    first = values[0]

    for value in values:
        if value != first:
            raise RuntimeError(
                "Different num_tasks detected between methods"
            )

    print()
    print(f"[CHECK] all methods used {first} tasks")


# ============================================================
# Speedup
# ============================================================

def calculate_speedup(results):
    baseline = next(
        result
        for result in results
        if result["method"] == "baseline"
    )

    baseline_time = baseline["gpu_submit_wait_ms"]

    if baseline_time <= 0.0:
        raise RuntimeError(
            "Baseline gpu_submit_wait_ms must be > 0"
        )

    for result in results:
        current_time = result["gpu_submit_wait_ms"]

        if current_time <= 0.0:
            result["speedup"] = 0.0
            result["reduction_percent"] = 0.0
            continue

        result["speedup"] = baseline_time / current_time

        result["reduction_percent"] = (
            (baseline_time - current_time)
            / baseline_time
            * 100.0
        )


# ============================================================
# 1STG 実行時間グラフ
# ============================================================

def plot_execution_time(stg_name, results, stg_label=None):
    labels = [result["display"] for result in results]
    times = [
        result["gpu_submit_wait_ms"]
        for result in results
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, times)

    plt.xlabel("Method")
    plt.ylabel("GPU Submit Wait Time [ms]")

    if stg_label is None:
        stg_label = stg_name

    plt.title(f"Execution Time - {stg_label}")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_path = (
        ROOT / f"{stg_name}_gpu_submit_wait.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Graph: {output_path}")


# ============================================================
# 1STG Speedupグラフ
# ============================================================

def plot_speedup(stg_name, results, stg_label=None):
    labels = [result["display"] for result in results]
    speedups = [result["speedup"] for result in results]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, speedups)

    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Method")
    plt.ylabel("Speedup [x]")

    if stg_label is None:
        stg_label = stg_name

    plt.title(f"Speedup - {stg_label}")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, speedups):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}x",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_path = ROOT / f"{stg_name}_speedup.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Graph: {output_path}")


# ============================================================
# 1STG 結果表示
# ============================================================

def print_stg_results(stg_name, results):
    print()
    print("=" * 90)
    print(f"Final Comparison: {stg_name}")
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
# 全STG 実行時間グラフ
# ============================================================

def plot_all_execution_times(all_stg_results):
    stg_names = list(all_stg_results.keys())

    stg_labels = [
        STG_DISPLAY_NAMES.get(name, name)
        for name in stg_names
    ]

    x = np.arange(len(stg_names))
    width = 0.18

    plt.figure(figsize=(13, 6))

    for method_index, method_cfg in enumerate(METHODS):
        values = []

        for stg_name in stg_names:
            result = next(
                result
                for result in all_stg_results[stg_name]
                if result["method"] == method_cfg["method"]
            )

            values.append(
                result["gpu_submit_wait_ms"]
            )

        offset = (
            method_index
            - (len(METHODS) - 1) / 2
        ) * width

        bars = plt.bar(
            x + offset,
            values,
            width,
            label=method_cfg["display"],
        )

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xlabel("STG")
    plt.ylabel("GPU Submit Wait Time [ms]")
    plt.title("Execution Time Comparison Across STGs")

    plt.xticks(
        x,
        stg_labels,
        rotation=20,
        ha="right",
    )

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = (
        ROOT / "all_stg_execution_time_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Graph: {output_path}")


# ============================================================
# 全STG Speedupグラフ
# ============================================================

def plot_all_speedups(all_stg_results):
    stg_names = list(all_stg_results.keys())

    stg_labels = [
        STG_DISPLAY_NAMES.get(name, name)
        for name in stg_names
    ]

    x = np.arange(len(stg_names))
    width = 0.18

    plt.figure(figsize=(13, 6))

    for method_index, method_cfg in enumerate(METHODS):
        values = []

        for stg_name in stg_names:
            result = next(
                result
                for result in all_stg_results[stg_name]
                if result["method"] == method_cfg["method"]
            )

            values.append(result["speedup"])

        offset = (
            method_index
            - (len(METHODS) - 1) / 2
        ) * width

        bars = plt.bar(
            x + offset,
            values,
            width,
            label=method_cfg["display"],
        )

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("STG")
    plt.ylabel("Speedup [x]")
    plt.title("Speedup Comparison Across STGs")

    plt.xticks(
        x,
        stg_labels,
        rotation=20,
        ha="right",
    )

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = (
        ROOT / "all_stg_speedup_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Graph: {output_path}")


# ============================================================
# 全STG 通常評価結果表示
# ============================================================

def print_all_results(all_stg_results):
    print()
    print()
    print("#" * 95)
    print("# ALL STG EXECUTION TIME / SPEEDUP RESULTS")
    print("#" * 95)

    for stg_name, results in all_stg_results.items():
        print()

        stg_label = STG_DISPLAY_NAMES.get(
            stg_name,
            stg_name,
        )

        print(f"[{stg_label}]")

        for result in results:
            print(
                f"  "
                f"{result['display']:15s} "
                f"{result['gpu_submit_wait_ms']:10.3f} ms"
                f"   "
                f"{result['speedup']:7.3f}x"
                f"   "
                f"{result['reduction_percent']:7.2f}%"
            )


# ============================================================
# PHASE 2:
# Nsight Systems で SMs Active を測定
#
# 重要:
#   実行時間測定とは完全に別実行
#   各 STG × 各手法をもう一度1回ずつ実行する
# ============================================================

def check_nsys():
    if NSYS is None:
        raise RuntimeError(
            "nsys command was not found in PATH"
        )

    print()
    print("=" * 80)
    print("Nsight Systems")
    print("=" * 80)
    print(f"nsys                 : {NSYS}")
    print(
        f"GPU metrics device   : "
        f"{GPU_METRICS_DEVICE}"
    )
    print(
        f"GPU metrics frequency: "
        f"{GPU_METRICS_FREQUENCY} Hz"
    )
    print("=" * 80)


def query_sm_active_samples(sqlite_path):
    """
    Nsight Systems SQLite exportから
    SMs Active の時系列サンプルを取得する。

    各 main は対象手法だけを実行する前提。

    評価区間:
      対象手法で最初に SMs Active > 0 となったサンプルから
      最後に SMs Active > 0 となったサンプルまで。

    前後の待機中の 0% は除外する。
    一方、手法の実行区間内に現れる 0% は除外せず平均する。
    したがって、逐次実行など別手法のSM使用率は混ざらない。
    """

    connection = sqlite3.connect(str(sqlite_path))

    try:
        cursor = connection.cursor()

        rows = cursor.execute(
            """
            SELECT
                g.timestamp,
                g.value
            FROM GPU_METRICS AS g
            JOIN TARGET_INFO_GPU_METRICS AS info
              USING (metricId)
            WHERE info.metricName LIKE 'SMs Active%'
            ORDER BY g.timestamp
            """
        ).fetchall()

    finally:
        connection.close()

    if not rows:
        raise RuntimeError(
            "SMs Active samples were not found in "
            f"{sqlite_path}"
        )

    timestamps = [int(row[0]) for row in rows]
    values = [float(row[1]) for row in rows]

    active_indices = [
        i
        for i, value in enumerate(values)
        if value > 0.0
    ]

    if not active_indices:
        raise RuntimeError(
            "All SMs Active samples were 0%"
        )

    first_index = active_indices[0]
    last_index = active_indices[-1]

    workload_timestamps = timestamps[
        first_index:last_index + 1
    ]

    workload_values = values[
        first_index:last_index + 1
    ]

    return workload_timestamps, workload_values


def measure_sm_active_for_stg(
    method_cfg,
    input_file,
):
    """
    1手法 × 1STGをNsight Systemsで1回実行し、
    その手法だけのSMs Active [%]平均を返す。

    前提:
      STG/main                  -> Sequentialのみ
      STG_existing_method/main  -> Existingのみ
      STG_existing_method_GC/main -> Existing + GCのみ
      STG_my_method/main        -> Proposedのみ
    """

    method = method_cfg["method"]
    method_dir = ROOT / method_cfg["dir"]
    binary_path = method_dir / method_cfg["binary"]

    print()
    print("-" * 80)
    print("SM ACTIVE PROFILING")
    print(f"STG    : {input_file.name}")
    print(f"Method : {method}")
    print("-" * 80)

    env = dict(os.environ)
    env["TMPDIR"] = str(NSYS_TMP_ROOT)

    report_prefix = (
        NSYS_RESULT_DIR
        / f"{input_file.stem}_{method}"
    )
    report_path = Path(str(report_prefix) + ".nsys-rep")
    sqlite_path = Path(str(report_prefix) + ".sqlite")

    if report_path.exists():
        report_path.unlink()

    if sqlite_path.exists():
        sqlite_path.unlink()

    profile_cmd = [
        NSYS,
        "profile",
        "--force-overwrite=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace=cuda",
        f"--gpu-metrics-devices={GPU_METRICS_DEVICE}",
        f"--gpu-metrics-frequency={GPU_METRICS_FREQUENCY}",
        f"--output={report_prefix}",
        binary_path,
        input_file,
    ]

    run_command(
        profile_cmd,
        method_dir,
        env=env,
    )

    if not report_path.exists():
        raise RuntimeError(
            f"Nsight report not found: {report_path}"
        )

    export_cmd = [
        NSYS,
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        f"--output={sqlite_path}",
        report_path,
    ]

    run_command(
        export_cmd,
        method_dir,
        env=env,
    )

    if not sqlite_path.exists():
        raise RuntimeError(
            f"Nsight SQLite export not found: {sqlite_path}"
        )

    timestamps, values = query_sm_active_samples(
        sqlite_path
    )

    avg_sm_active = mean(values)
    min_sm_active = min(values)
    max_sm_active = max(values)

    duration_ms = (
        timestamps[-1] - timestamps[0]
    ) / 1_000_000.0

    print()
    print(
        f"[SM ACTIVE RESULT] "
        f"{input_file.stem} / {method}"
    )
    print(
        f"  samples         = {len(values)}"
    )
    print(
        f"  measured window = {duration_ms:.3f} ms"
    )
    print(
        f"  SMs Active avg  = {avg_sm_active:.2f}%"
    )
    print(
        f"  SMs Active min  = {min_sm_active:.2f}%"
    )
    print(
        f"  SMs Active max  = {max_sm_active:.2f}%"
    )
    print(
        f"  report          = {report_path}"
    )
    print(
        f"  sqlite          = {sqlite_path}"
    )

    return {
        "stg": input_file.stem,
        "stg_file": input_file.name,
        "method": method,
        "display": method_cfg["display"],
        "sm_active_pct": avg_sm_active,
        "sm_active_min_pct": min_sm_active,
        "sm_active_max_pct": max_sm_active,
        "sm_active_samples": len(values),
        "sm_active_window_ms": duration_ms,
        "nsys_report": str(report_path),
        "sqlite": str(sqlite_path),
    }


# ============================================================
# 全STG × 全手法 SM Active測定
# ============================================================

def measure_all_sm_active():
    print()
    print()
    print("#" * 95)
    print("# PHASE 2: SMs Active Measurement")
    print("#" * 95)
    print(
        "# Execution time / Speedup measurement is already finished."
    )
    print(
        "# Each method-only binary is executed once under Nsight Systems."
    )
    print("#" * 95)

    check_nsys()

    all_sm_results = {}

    for input_file in INPUT_FILES:
        stg_key = input_file.stem
        stg_label = STG_DISPLAY_NAMES.get(
            stg_key,
            stg_key,
        )

        print()
        print()
        print("#" * 90)
        print(f"# SM ACTIVE STG: {stg_label}")
        print("#" * 90)

        results = []

        for method_cfg in METHODS:
            result = measure_sm_active_for_stg(
                method_cfg,
                input_file,
            )

            results.append(result)

        all_sm_results[stg_key] = results

    return all_sm_results


# ============================================================
# 全STG SM Activeグラフ
# ============================================================

def plot_all_sm_active(all_sm_results):
    stg_names = list(all_sm_results.keys())

    stg_labels = [
        STG_DISPLAY_NAMES.get(name, name)
        for name in stg_names
    ]

    x = np.arange(len(stg_names))
    width = 0.18

    plt.figure(figsize=(13, 6))

    for method_index, method_cfg in enumerate(METHODS):
        values = []

        for stg_name in stg_names:
            result = next(
                result
                for result in all_sm_results[stg_name]
                if result["method"] == method_cfg["method"]
            )

            values.append(
                result["sm_active_pct"]
            )

        offset = (
            method_index
            - (len(METHODS) - 1) / 2
        ) * width

        bars = plt.bar(
            x + offset,
            values,
            width,
            label=method_cfg["display"],
        )

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xlabel("STG")
    plt.ylabel("Average SMs Active [%]")
    plt.title("SMs Active Comparison Across STGs")

    plt.xticks(
        x,
        stg_labels,
        rotation=20,
        ha="right",
    )

    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = (
        ROOT / "all_stg_sm_active_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print()
    print(f"Graph: {output_path}")


# ============================================================
# SM Active結果表示
# ============================================================

def print_all_sm_active(all_sm_results):
    print()
    print()
    print("#" * 95)
    print("# ALL STG SMs ACTIVE RESULTS")
    print("#" * 95)

    for stg_name, results in all_sm_results.items():
        stg_label = STG_DISPLAY_NAMES.get(
            stg_name,
            stg_name,
        )

        print()
        print(f"[{stg_label}]")

        for result in results:
            print(
                f"  "
                f"{result['display']:15s} "
                f"{result['sm_active_pct']:8.2f}%"
                f"   "
                f"samples={result['sm_active_samples']:5d}"
                f"   "
                f"window={result['sm_active_window_ms']:.3f} ms"
            )


# ============================================================
# main
# ============================================================

def main():

    # ========================================================
    # 初期確認・コンパイル
    # ========================================================

    check_input_files()
    build_all_methods()

    # ========================================================
    # PHASE 1
    # 通常実行で実行時間とSpeedupを測る
    # ========================================================

    all_stg_results = {}

    print()
    print()
    print("#" * 95)
    print("# PHASE 1: Execution Time / Speedup")
    print("#" * 95)

    for input_file in INPUT_FILES:
        stg_key = input_file.stem
        stg_label = STG_DISPLAY_NAMES.get(
            stg_key,
            stg_key,
        )

        print()
        print()
        print("#" * 90)
        print(f"# STG: {stg_label}")
        print("#" * 90)

        results = []

        for method_cfg in METHODS:
            result = run_method_for_stg(
                method_cfg,
                input_file,
            )

            results.append(result)

        validate_num_tasks(results)
        calculate_speedup(results)

        print_stg_results(
            stg_label,
            results,
        )

        plot_execution_time(
            stg_key,
            results,
            stg_label=stg_label,
        )

        plot_speedup(
            stg_key,
            results,
            stg_label=stg_label,
        )

        all_stg_results[stg_key] = results

    # ========================================================
    # PHASE 1の全体結果を出力
    # ここまでNsight Systemsは使用しない
    # ========================================================

    print_all_results(all_stg_results)

    plot_all_execution_times(
        all_stg_results
    )

    plot_all_speedups(
        all_stg_results
    )

    # ========================================================
    # PHASE 2
    # 実行時間・Speedupの図を出した後、
    # 全5 STG × 全4手法をもう一度実行し、
    # Nsight SystemsでSMs Activeを測定
    # ========================================================

    all_sm_results = measure_all_sm_active()

    print_all_sm_active(
        all_sm_results
    )

    plot_all_sm_active(
        all_sm_results
    )

    # ========================================================
    # 終了
    # ========================================================

    print()
    print("=" * 90)
    print("All evaluations completed.")
    print("=" * 90)

    print()
    print("PHASE 1")
    print(
        "  Execution time : gpu_submit_wait_ms"
    )
    print(
        "  Speedup        : "
        "Baseline time / Method time"
    )
    print(
        f"  Runs           : {RUNS}"
    )
    print(
        f"  Warmup         : {WARMUP_RUNS}"
    )
    print(
        f"  Average        : "
        f"{RUNS - WARMUP_RUNS} runs"
    )

    print()
    print("PHASE 2")
    print(
        "  Utilization    : Average SMs Active [%]"
    )
    print(
        "  Profiling      : "
        "1 separate method-only Nsight Systems run "
        "per STG × Method"
    )
    print(
        "  Range          : "
        "first active sample -> last active sample"
    )
    print(
        "  Internal 0%    : included"
    )
    print(
        f"  Sampling       : "
        f"{GPU_METRICS_FREQUENCY} Hz"
    )
    print(
        f"  Reports        : {NSYS_RESULT_DIR}"
    )

    print()
    print("Generated summary PNG files:")
    print(
        "  all_stg_execution_time_comparison.png"
    )
    print(
        "  all_stg_speedup_comparison.png"
    )
    print(
        "  all_stg_sm_active_comparison.png"
    )
    print()


if __name__ == "__main__":
    main()