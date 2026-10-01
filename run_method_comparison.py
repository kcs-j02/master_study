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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# 基本設定
# ============================================================

ROOT = Path(__file__).resolve().parent

# グラフ保存先
COMPARISON_FIGURE_DIR = ROOT / "comparison_figures"
COMPARISON_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

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




def format_stg_label_for_axis(label):
    label_map = {
        "Fully Parallel": "Fully\nParallel",
        "Multiple Long Branches": "Multiple Long\nBranches",
        "Mixed: Chain+Parallel ": "Mixed:\nChain+Parallel",
    }
    return label_map.get(label, label)


def get_speedup_label_offset(method_index, point_index, value, total_points):
    base_offsets = [10, 18, 26, 34]
    dx_offsets = [-6, -2, 2, 6]

    dy = base_offsets[method_index % len(base_offsets)]
    dx = dx_offsets[method_index % len(dx_offsets)]

    if value <= 1.15:
        dy += 6

    if point_index == 0:
        dx += 6
    elif point_index == total_points - 1:
        dx -= 6

    return dx, dy

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
        "display": "SEQ",
        "color": "C0",
        "dir": "STG",
        "binary": "main",
    },
    {
        "method": "existing_method",
        "display": "KS",
        "color": "C1",
        "dir": "STG_existing_method",
        "binary": "main",
    },
    {
        "method": "existing_method_gc",
        "display": "KS+GC",
        "color": "C2",
        "dir": "STG_existing_method_GC",
        "binary": "main",
    },
    {
        "method": "proposed",
        "display": "PR",
        "color": "C3",
        "dir": "STG_my_method",
        "binary": "main",
    },
]

METHOD_COLOR_MAP = {
    method_cfg["method"]: method_cfg["color"]
    for method_cfg in METHODS
}

METHOD_HATCH_MAP = {
    "baseline": "",
    "existing_method": "///",
    "existing_method_gc": "xxx",
    "proposed": "...",
}

METHOD_LINESTYLE_MAP = {
    "baseline": "-",
    "existing_method": "--",
    "existing_method_gc": "-.",
    "proposed": ":",
}

METHOD_MARKER_MAP = {
    "baseline": "o",
    "existing_method": "s",
    "existing_method_gc": "^",
    "proposed": "D",
}

METHOD_FILL_COLOR_MAP = {
    "baseline": "#f2f2f2",
    "existing_method": "#cfe2f3",
    "existing_method_gc": "#d9ead3",
    "proposed": "#f4cccc",
}

BAR_EDGE_COLOR = "black"
LINE_COLOR = "#333333"

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15
VALUE_FONTSIZE = 15
SMALL_VALUE_FONTSIZE = 13


def style_bar_container(bar_container, method):
    for bar in bar_container:
        bar.set_facecolor(METHOD_FILL_COLOR_MAP[method])
        bar.set_edgecolor(BAR_EDGE_COLOR)
        bar.set_linewidth(1.5)
        bar.set_hatch(METHOD_HATCH_MAP[method])


def method_patch_handles():
    handles = []

    for method_cfg in METHODS:
        handles.append(
            Patch(
                facecolor=METHOD_FILL_COLOR_MAP[method_cfg["method"]],
                edgecolor=BAR_EDGE_COLOR,
                linewidth=1.5,
                hatch=METHOD_HATCH_MAP[method_cfg["method"]],
                label=method_cfg["display"],
            )
        )

    return handles


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
            f"gpu_submit_wait_ms = {submit_wait_ms:.3g} ms",
            end="",
        )

        if kernel_ms is not None:
            print(
                f"  gpu_kernel_ms = {kernel_ms:.3g} ms"
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
        f"{avg_submit_wait_ms:.3g} ms"
    )
    print(
        f"  gpu_submit_wait_ms min     = "
        f"{min_submit_wait_ms:.3g} ms"
    )
    print(
        f"  gpu_submit_wait_ms max     = "
        f"{max_submit_wait_ms:.3g} ms"
    )

    if avg_kernel_ms is not None:
        print(
            f"  gpu_kernel_ms average      = "
            f"{avg_kernel_ms:.3g} ms"
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
    methods = [result["method"] for result in results]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_containers = []

    for xi, time_value, method in zip(x, times, methods):
        bars = ax.bar(
            xi,
            time_value,
            width=0.6,
            zorder=2,
        )
        style_bar_container(bars, method)
        bar_containers.append(bars)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Method", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Makespan [ms]", fontsize=LABEL_FONTSIZE)

    if stg_label is None:
        stg_label = stg_name

    ax.set_title(
        f"Execution Time - {stg_label}",
        fontsize=TITLE_FONTSIZE,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
        zorder=0,
    )

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.margins(x=0.08)

    for bars, value in zip(bar_containers, times):
        bar = bars.patches[0]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=VALUE_FONTSIZE,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
                pad=1.6,
            ),
            zorder=11,
            clip_on=False,
        )

    time_max = max(times) if times else 1.0
    ax.set_ylim(0, time_max * 1.30)

    ax.legend(
        method_patch_handles(),
        [cfg["display"] for cfg in METHODS],
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        ncol=2,
    )

    fig.tight_layout()

    output_path = (
        COMPARISON_FIGURE_DIR / f"{stg_name}_gpu_submit_wait.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Graph: {output_path}")

# ============================================================
# 1STG Speedupグラフ
# ============================================================

def plot_speedup(stg_name, results, stg_label=None):
    labels = [result["display"] for result in results]
    speedups = [result["speedup"] for result in results]
    methods = [result["method"] for result in results]

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    bars = []

    for xi, (value, method) in enumerate(zip(speedups, methods)):
        bar_container = ax.bar(
            xi,
            value,
            width=0.6,
            zorder=2,
        )
        style_bar_container(bar_container, method)
        bars.append(bar_container.patches[0])

    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
        color="#666666",
    )

    plt.xlabel("Method", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Speedup [x]", fontsize=LABEL_FONTSIZE)

    if stg_label is None:
        stg_label = stg_name

    plt.title(f"Speedup - {stg_label}", fontsize=TITLE_FONTSIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    plt.xticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    for bar, value in zip(bars, speedups):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2g}x",
            ha="center",
            va="bottom",
            fontsize=VALUE_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.4),
        )

    plt.ylim(0, max(1.2, max(speedups) * 1.35))
    plt.tight_layout()

    output_path = COMPARISON_FIGURE_DIR / f"{stg_name}_speedup.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Graph: {output_path}")


# ============================================================
# 1STG SM Activeグラフ
# ============================================================

def plot_sm_active(stg_name, results, stg_label=None):
    labels = [result["display"] for result in results]
    sm_active_values = [result["sm_active_pct"] for result in results]
    methods = [result["method"] for result in results]

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    bars = []

    for xi, (value, method) in enumerate(zip(sm_active_values, methods)):
        bar_container = ax.bar(
            xi,
            value,
            width=0.6,
            zorder=2,
        )
        style_bar_container(bar_container, method)
        bars.append(bar_container.patches[0])

    plt.xlabel("Method", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Average SMs Active [%]", fontsize=LABEL_FONTSIZE)

    if stg_label is None:
        stg_label = stg_name

    plt.title(f"SMs Active - {stg_label}", fontsize=TITLE_FONTSIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    plt.xticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    for bar, value in zip(bars, sm_active_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=VALUE_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.4),
        )

    plt.ylim(0, max(sm_active_values) * 1.18)
    plt.tight_layout()

    output_path = COMPARISON_FIGURE_DIR / f"{stg_name}_sm_active.png"

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
            f"{result['gpu_submit_wait_ms']:16.3g}"
            f"{result['speedup']:13.2g}x"
            f"{result['reduction_percent']:15.2f}%"
        )

    print("=" * 90)


# ============================================================
# 全STG 実行時間グラフ
# ============================================================

def plot_all_execution_time_and_speedup(all_stg_results):
    stg_names = [
        name
        for name in all_stg_results.keys()
        if name != "sample_mixed_chain_parallel"
    ]

    stg_labels = [
        format_stg_label_for_axis(
            STG_DISPLAY_NAMES.get(name, name)
        )
        for name in stg_names
    ]

    x = np.arange(len(stg_names))
    width = 0.18
    method_offsets = [
        (method_index - (len(METHODS) - 1) / 2) * width
        for method_index in range(len(METHODS))
    ]

    fig, ax_speedup = plt.subplots(figsize=(14, 6.5))
    ax_time = ax_speedup.twinx()

    ax_speedup.set_zorder(3)
    ax_time.set_zorder(2)
    ax_speedup.patch.set_alpha(0.0)

    speedup_by_stg = {
        stg_name: []
        for stg_name in stg_names
    }
    positions_by_stg = {
        stg_name: []
        for stg_name in stg_names
    }

    for method_index, method_cfg in enumerate(METHODS):
        time_values = []

        for stg_name in stg_names:
            result = next(
                result
                for result in all_stg_results[stg_name]
                if result["method"] == method_cfg["method"]
            )

            time_values.append(result["gpu_submit_wait_ms"])
            speedup_by_stg[stg_name].append(result["speedup"])

        bar_positions = x + method_offsets[method_index]

        bars = ax_time.bar(
            bar_positions,
            time_values,
            width,
            zorder=1,
        )
        style_bar_container(bars, method_cfg["method"])

        for stg_name, xpos in zip(stg_names, bar_positions):
            positions_by_stg[stg_name].append(xpos)

        for bar, value in zip(bars, time_values):
            ax_time.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=SMALL_VALUE_FONTSIZE,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.92,
                    pad=1.3,
                ),
                zorder=11,
                clip_on=False,
            )

    first_line = None

    for stg_index, stg_name in enumerate(stg_names):
        line = ax_speedup.plot(
            positions_by_stg[stg_name],
            speedup_by_stg[stg_name],
            color=LINE_COLOR,
            marker="o",
            linewidth=2.4,
            markersize=6.5,
            zorder=10,
        )

        if first_line is None:
            first_line = line

        for method_index, (xpos, value) in enumerate(
            zip(positions_by_stg[stg_name], speedup_by_stg[stg_name])
        ):
            dx, dy = get_speedup_label_offset(
                method_index,
                method_index,
                value,
                len(METHODS),
            )
            ax_speedup.annotate(
                f"{value:.2g}x",
                xy=(xpos, value),
                xytext=(dx, dy + 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=SMALL_VALUE_FONTSIZE,
                color="black",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.95,
                    pad=1.3,
                ),
                zorder=12,
                clip_on=False,
            )

    ax_speedup.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
        color="#666666",
        alpha=0.8,
        zorder=2,
    )

    ax_speedup.set_xlabel("STG", fontsize=LABEL_FONTSIZE)
    ax_speedup.set_ylabel("Speedup [x]", fontsize=LABEL_FONTSIZE)
    ax_time.set_ylabel("GPU Submit Wait Time [ms]", fontsize=LABEL_FONTSIZE)
    ax_speedup.set_title(
        "Execution Time and Speedup Comparison Across STGs",
        fontsize=TITLE_FONTSIZE,
    )

    ax_speedup.set_xticks(x)
    ax_speedup.set_xticklabels(stg_labels)

    ax_speedup.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax_time.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_speedup.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_speedup.margins(x=0.05)

    time_max = max(
        result["gpu_submit_wait_ms"]
        for results in all_stg_results.values()
        for result in results
        if result["stg"] != "sample_mixed_chain_parallel"
    )
    speedup_max = max(
        result["speedup"]
        for results in all_stg_results.values()
        for result in results
        if result["stg"] != "sample_mixed_chain_parallel"
    )

    ax_time.set_ylim(0, time_max * 1.32)
    ax_speedup.set_ylim(0, max(1.2, speedup_max * 1.65))

    ax_speedup.legend(
        first_line,
        ["Speedup"],
        loc="upper left",
        fontsize=LEGEND_FONTSIZE,
    )

    ax_time.legend(
        method_patch_handles(),
        [cfg["display"] for cfg in METHODS],
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        ncol=2,
    )

    plt.tight_layout()

    output_path = (
        COMPARISON_FIGURE_DIR / "all_stg_execution_time_speedup_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Graph: {output_path}")


# ============================================================
# 全STG 分割グラフ
# ============================================================

def get_all_comparison_stg_names(results_by_stg):
    return [
        name
        for name in results_by_stg.keys()
        if name != "sample_mixed_chain_parallel"
    ]


def plot_all_grouped_metric(
    all_stg_results,
    result_key,
    ylabel,
    title,
    filename,
    value_format,
):
    stg_names = get_all_comparison_stg_names(all_stg_results)
    stg_labels = [
        format_stg_label_for_axis(STG_DISPLAY_NAMES.get(name, name))
        for name in stg_names
    ]
    x = np.arange(len(stg_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 6.5))

    for method_index, method_cfg in enumerate(METHODS):
        values = []
        for stg_name in stg_names:
            result = next(
                result
                for result in all_stg_results[stg_name]
                if result["method"] == method_cfg["method"]
            )
            values.append(result[result_key])

        offset = (method_index - (len(METHODS) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, zorder=2)
        style_bar_container(bars, method_cfg["method"])

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                value_format.format(value),
                ha="center",
                va="bottom",
                fontsize=SMALL_VALUE_FONTSIZE,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.92,
                    pad=1.3,
                ),
                zorder=3,
                clip_on=False,
            )

    ax.set_xlabel("STG", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(stg_labels)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.margins(x=0.05)
    ax.legend(
        method_patch_handles(),
        [cfg["display"] for cfg in METHODS],
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        ncol=2,
    )

    if result_key == "speedup":
        ax.axhline(
            y=1.0,
            linestyle="--",
            linewidth=1,
            color="#666666",
            alpha=0.8,
        )
        max_value = max(
            result[result_key]
            for results in all_stg_results.values()
            for result in results
            if result["stg"] != "sample_mixed_chain_parallel"
        )
        ax.set_ylim(0, max(1.2, max_value * 1.25))

    plt.tight_layout()
    output_path = COMPARISON_FIGURE_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph: {output_path}")


def plot_all_execution_time_split(all_stg_results):
    plot_all_grouped_metric(
        all_stg_results,
        "gpu_submit_wait_ms",
        "GPU Submit Wait Time [ms]",
        "Execution Time Comparison Across STGs",
        "all_stg_execution_time_comparison.png",
        "{:.3g}",
    )


def plot_all_speedup_split(all_stg_results):
    plot_all_grouped_metric(
        all_stg_results,
        "speedup",
        "Speedup [x]",
        "Speedup Comparison Across STGs",
        "all_stg_speedup_comparison.png",
        "{:.2g}x",
    )


def plot_execution_time_vs_sm_active(all_stg_results, all_sm_results):
    stg_names = get_all_comparison_stg_names(all_stg_results)
    column_count = 2
    row_count = (len(stg_names) + column_count - 1) // column_count
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(14, 10),
        sharey=True,
        squeeze=False,
    )

    all_sm_values = [
        result["sm_active_pct"]
        for stg_name in stg_names
        for result in all_sm_results[stg_name]
    ]
    shared_y_min = max(0.0, min(all_sm_values) - 7.0)
    shared_y_max = min(105.0, max(all_sm_values) + 5.0)

    label_offsets = {
        "baseline": (-8, 10),
        "existing_method": (8, 10),
        "existing_method_gc": (8, -16),
        "proposed": (8, 10),
    }

    for ax, stg_name in zip(axes.flat, stg_names):
        sm_by_method = {
            result["method"]: result["sm_active_pct"]
            for result in all_sm_results[stg_name]
        }

        execution_times = []
        sm_values = []

        for method_cfg in METHODS:
            time_result = next(
                result
                for result in all_stg_results[stg_name]
                if result["method"] == method_cfg["method"]
            )

            method = method_cfg["method"]
            execution_time = time_result["gpu_submit_wait_ms"]
            sm_active = sm_by_method[method]
            execution_times.append(execution_time)
            sm_values.append(sm_active)

            ax.scatter(
                execution_time,
                sm_active,
                s=125,
                color=METHOD_COLOR_MAP[method],
                marker=METHOD_MARKER_MAP[method],
                edgecolors="black",
                linewidths=1.0,
                zorder=3,
            )

            dx, dy = label_offsets[method]
            ax.annotate(
                method_cfg["display"],
                xy=(execution_time, sm_active),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="right" if dx < 0 else "left",
                va="bottom" if dy >= 0 else "top",
                fontsize=11,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.82,
                    pad=0.8,
                ),
                zorder=4,
            )

        execution_times = np.asarray(execution_times, dtype=float)
        sm_values = np.asarray(sm_values, dtype=float)
        correlation = float(np.corrcoef(execution_times, sm_values)[0, 1])

        if np.ptp(execution_times) > 0.0:
            slope, intercept = np.polyfit(execution_times, sm_values, 1)
            fit_x = np.linspace(
                execution_times.min(),
                execution_times.max(),
                100,
            )
            ax.plot(
                fit_x,
                slope * fit_x + intercept,
                color="#555555",
                linestyle="--",
                linewidth=1.8,
                alpha=0.85,
                zorder=2,
            )

        x_padding = max(np.ptp(execution_times) * 0.12, 5.0)
        ax.set_xlim(
            execution_times.min() - x_padding,
            execution_times.max() + x_padding,
        )
        ax.set_ylim(shared_y_min, shared_y_max)
        ax.set_title(
            STG_DISPLAY_NAMES.get(stg_name, stg_name).strip(),
            fontsize=17,
        )
        ax.text(
            0.04,
            0.06,
            f"Pearson r = {correlation:.2f}",
            transform=ax.transAxes,
            fontsize=13,
            bbox=dict(
                facecolor="white",
                edgecolor="#777777",
                alpha=0.9,
                pad=3.0,
            ),
            zorder=5,
        )
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)

    for ax in axes.flat[len(stg_names):]:
        ax.set_visible(False)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKER_MAP[method_cfg["method"]],
            color=METHOD_COLOR_MAP[method_cfg["method"]],
            markeredgecolor="black",
            markeredgewidth=1.0,
            linestyle="None",
            markersize=9,
            label=method_cfg["display"],
        )
        for method_cfg in METHODS
    ]

    fig.suptitle(
        "Execution Time vs. SM Utilization by STG",
        fontsize=TITLE_FONTSIZE,
        y=0.98,
    )
    fig.supxlabel(
        "GPU Submit Wait Time [ms]",
        fontsize=LABEL_FONTSIZE,
        y=0.035,
    )
    fig.supylabel(
        "Average SMs Active [%]",
        fontsize=LABEL_FONTSIZE,
        x=0.035,
    )
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        fontsize=LEGEND_FONTSIZE,
        ncol=len(METHODS),
        frameon=True,
    )

    fig.tight_layout(rect=(0.055, 0.055, 1.0, 0.90))
    output_path = (
        COMPARISON_FIGURE_DIR
        / "all_stg_execution_time_sm_active_comparison.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
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
                f"{result['gpu_submit_wait_ms']:10.3g} ms"
                f"   "
                f"{result['speedup']:7.2g}x"
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
    stg_names = [
        name
        for name in all_sm_results.keys()
        if name != "sample_mixed_chain_parallel"
    ]

    stg_labels = [
        format_stg_label_for_axis(
            STG_DISPLAY_NAMES.get(name, name)
        )
        for name in stg_names
    ]

    x = np.arange(len(stg_names))
    width = 0.18

    plt.figure(figsize=(14, 6.5))
    ax = plt.gca()

    for method_index, method_cfg in enumerate(METHODS):
        values = []

        for stg_name in stg_names:
            result = next(
                result
                for result in all_sm_results[stg_name]
                if result["method"] == method_cfg["method"]
            )
            values.append(result["sm_active_pct"])

        offset = (
            method_index - (len(METHODS) - 1) / 2
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            zorder=2,
        )
        style_bar_container(bars, method_cfg["method"])

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=SMALL_VALUE_FONTSIZE,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.92,
                    pad=1.3,
                ),
                zorder=3,
                clip_on=False,
            )

    ax.set_xlabel("STG", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Average SMs Active [%]", fontsize=LABEL_FONTSIZE)
    ax.set_title(
        "SMs Active Comparison Across STGs",
        fontsize=TITLE_FONTSIZE,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stg_labels)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_ylim(0, 103)
    ax.margins(x=0.04)

    ax.legend(
        method_patch_handles(),
        [cfg["display"] for cfg in METHODS],
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        ncol=2,
    )

    plt.tight_layout()

    output_path = (
        COMPARISON_FIGURE_DIR / "all_stg_sm_active_comparison.png"
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

    plot_all_execution_time_split(all_stg_results)
    plot_all_speedup_split(all_stg_results)

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

    plot_execution_time_vs_sm_active(
        all_stg_results,
        all_sm_results,
    )

    mixed_stg_key = "sample_mixed_chain_parallel"
    mixed_stg_label = STG_DISPLAY_NAMES.get(
        mixed_stg_key,
        mixed_stg_key,
    )

    plot_sm_active(
        mixed_stg_key,
        all_sm_results[mixed_stg_key],
        stg_label=mixed_stg_label,
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
        "  all_stg_execution_time_sm_active_comparison.png"
    )
    print(
        "  all_stg_sm_active_comparison.png"
    )
    print(
        "  sample_mixed_chain_parallel_sm_active.png"
    )
    print()


if __name__ == "__main__":
    main()
