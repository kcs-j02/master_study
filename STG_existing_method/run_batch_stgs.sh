#!/usr/bin/env bash
set -euo pipefail


# ============================================================
# Directory
# ============================================================

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER_DIR="$(cd "$ROOT_DIR/.." && pwd)"

TASKFLOW_INC="/home/kobayashi/taskflow"

# Sequential
SEQUENTIAL_DIR="$MASTER_DIR/STG"
SEQUENTIAL_MAIN="$SEQUENTIAL_DIR/main"

# Existing Method
EM_DIR="$ROOT_DIR"
EM_MAIN="$EM_DIR/main"

# Nsight Systems
NSYS="/usr/local/cuda/bin/nsys"
NSYS_TMP_ROOT="$HOME/tmp/nsys"

# Nsight結果保存場所
NSYS_RESULT_DIR="$ROOT_DIR/nsys_reports"

mkdir -p "$NSYS_TMP_ROOT"
mkdir -p "$NSYS_RESULT_DIR"


# ============================================================
# Evaluation settings
# ============================================================

RUNS=12
WARMUP_RUNS=2


# ============================================================
# STG files
# ============================================================

STGS=(
  "$MASTER_DIR/sample_mixed_chain_parallel.stg"
  "$MASTER_DIR/sample_fully_parallel.stg"
  "$MASTER_DIR/sample_Multiple_long_branches.stg"
  "$MASTER_DIR/sample_random.stg"
  "$MASTER_DIR/sample_trial.stg"
)


# ============================================================
# Final result arrays
# ============================================================

RESULT_NAMES=()
RESULT_SEQ=()
RESULT_EM=()
RESULT_SPEEDUP=()
RESULT_SM=()


# ============================================================
# Build Sequential
# ============================================================

echo "Building Sequential..."

cd "$SEQUENTIAL_DIR"

rm -f main

nvcc -O2 -std=c++20 main.cu \
  -I"${TASKFLOW_INC}" \
  -o main

echo "Sequential build completed."


# ============================================================
# Build Existing Method
# ============================================================

echo "Building EM..."

cd "$EM_DIR"

rm -f main

nvcc -O2 -std=c++20 main.cu \
  -I"${TASKFLOW_INC}" \
  -o main

echo "EM build completed."
echo


# ============================================================
# Execution time measurement
#
# 12回実行
# 最初の2回 = Warmup
# 残り10回の gpu_submit_wait_ms を平均
# ============================================================

measure_average() {

  local executable="$1"
  local stg="$2"
  local method_name="$3"

  local sum="0"
  local count=0
  local logfile
  local time_ms
  local average


  for ((run=1; run<=RUNS; run++)); do

    logfile=$(mktemp)

    "$executable" "$stg" > "$logfile" 2>&1


    # --------------------------------------------------------
    # gpu_submit_wait_ms
    # --------------------------------------------------------

    time_ms=$(
      grep 'gpu_submit_wait_ms:' "$logfile" \
        | tail -n1 \
        | awk '{print $2}'
    )


    rm -f "$logfile"


    echo "[$method_name] run $run/$RUNS : ${time_ms} ms" >&2


    # --------------------------------------------------------
    # 最初の2回はWarmup
    # --------------------------------------------------------

    if (( run > WARMUP_RUNS )); then

      sum=$(
        awk \
          -v a="$sum" \
          -v b="$time_ms" \
          'BEGIN {
              printf "%.10f", a + b
          }'
      )

      ((count+=1))

    fi

  done


  # ----------------------------------------------------------
  # 残り10回平均
  # ----------------------------------------------------------

  average=$(
    awk \
      -v sum="$sum" \
      -v count="$count" \
      'BEGIN {
          printf "%.3f", sum / count
      }'
  )


  echo "$average"
}


# ============================================================
# SM Utilization measurement
#
# Existing MethodをNsight Systemsで別に1回実行
#
# main.cuでは
#
#   1. Sequential
#   2. Existing Method
#
# の順にGPU処理するため、
# 最初のSequential区間を除外する。
#
# .nsys-rep / .sqlite は保存する。
# ============================================================

measure_sm_utilization() {

  local stg="$1"

  local name
  local report_prefix
  local report_file
  local sqlite_file
  local sm_utilization


  name=$(basename "$stg" .stg)


  # ==========================================================
  # Save filenames
  # ==========================================================

  report_prefix="$NSYS_RESULT_DIR/${name}_em"

  report_file="${report_prefix}.nsys-rep"
  sqlite_file="${report_prefix}.sqlite"


  # 古い計測結果を削除
  rm -f "$report_file"
  rm -f "$sqlite_file"


  echo "[Nsight] profiling $name ..." >&2


  # ==========================================================
  # Nsight Systems profile
  # ==========================================================

  TMPDIR="$NSYS_TMP_ROOT" "$NSYS" profile \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --trace=cuda \
    --gpu-metrics-devices=0 \
    --gpu-metrics-frequency=10000 \
    --output="$report_prefix" \
    "$EM_MAIN" "$stg" \
    > /dev/null 2>&1


  # ==========================================================
  # nsys-rep -> SQLite
  # ==========================================================

  TMPDIR="$NSYS_TMP_ROOT" "$NSYS" export \
    --type=sqlite \
    --force-overwrite=true \
    --output="$sqlite_file" \
    "$report_file" \
    > /dev/null 2>&1


  # ==========================================================
  # SM Utilization計算
  # ==========================================================

  sm_utilization=$(
    python3 - "$sqlite_file" <<'PY'

import sqlite3
import sys


sqlite_file = sys.argv[1]


# ============================================================
# Open SQLite
# ============================================================

con = sqlite3.connect(sqlite_file)
cur = con.cursor()


# ============================================================
# Get SMs Active
# ============================================================

rows = cur.execute("""
SELECT
    g.timestamp,
    g.value
FROM GPU_METRICS AS g
JOIN TARGET_INFO_GPU_METRICS AS info
    USING (metricId)
WHERE info.metricName LIKE 'SMs Active%'
ORDER BY g.timestamp
""").fetchall()

con.close()


# ============================================================
# No data
# ============================================================

if not rows:
    print("N/A")
    sys.exit(0)


values = [float(row[1]) for row in rows]


# ============================================================
# Detect Active regions
#
# SMs Active > 0 が連続している区間を取得
# ============================================================

regions = []

start = None


for i, value in enumerate(values):

    if value > 0.0 and start is None:

        start = i


    elif value <= 0.0 and start is not None:

        regions.append(
            (start, i - 1)
        )

        start = None


# 最後までActiveの場合
if start is not None:

    regions.append(
        (start, len(values) - 1)
    )


# ============================================================
# Activeなし
# ============================================================

if len(regions) == 0:

    print("0.00")
    sys.exit(0)


# ============================================================
# Initial Sequential executionを除外
#
# regions[0]
#     ↓
# Sequential
#
# regions[1]
#     ↓
# Existing Method開始
# ============================================================

if len(regions) < 2:

    print("N/A")
    sys.exit(0)


# ============================================================
# Existing Methodの測定範囲
#
# 2番目のActive区間開始
#          ～
# 最後のActive区間終了
#
# Existing Method実行中の0%は平均に含める。
# ============================================================

target_start = regions[1][0]
target_end   = regions[-1][1]


target_values = values[
    target_start:
    target_end + 1
]


if not target_values:

    print("N/A")
    sys.exit(0)


# ============================================================
# Average SMs Active
# ============================================================

avg = sum(target_values) / len(target_values)


print(f"{avg:.2f}")

PY
  )


  echo "[Nsight] saved:" >&2
  echo "  $report_file" >&2
  echo "  $sqlite_file" >&2


  echo "$sm_utilization"
}


# ============================================================
# Header
# ============================================================

echo
echo "================================================================================================================================"
echo " Existing Method Evaluation"
echo "================================================================================================================================"

printf "%-35s %22s %20s %15s %20s\n" \
  "STG" \
  "Sequential Time [ms]" \
  "EM Time [ms]" \
  "Speedup [x]" \
  "SM Utilization [%]"

printf "%-35s %22s %20s %15s %20s\n" \
  "-----------------------------------" \
  "----------------------" \
  "--------------------" \
  "---------------" \
  "--------------------"


# ============================================================
# Evaluate each STG
# ============================================================

for stg in "${STGS[@]}"; do


  # ----------------------------------------------------------
  # STG存在確認
  # ----------------------------------------------------------

  if [ ! -f "$stg" ]; then

    echo "STG file not found: $stg" >&2
    continue

  fi


  name=$(basename "$stg" .stg)


  echo >&2
  echo "================================================================" >&2
  echo "STG: $name" >&2
  echo "================================================================" >&2


  # ==========================================================
  # Sequential
  #
  # 12 runs
  # 2 warmup
  # 残り10回平均
  # ==========================================================

  sequential_time=$(
    measure_average \
      "$SEQUENTIAL_MAIN" \
      "$stg" \
      "Sequential"
  )


  # ==========================================================
  # Existing Method
  #
  # 12 runs
  # 2 warmup
  # 残り10回平均
  # ==========================================================

  em_time=$(
    measure_average \
      "$EM_MAIN" \
      "$stg" \
      "EM"
  )


  # ==========================================================
  # Speedup
  #
  # Sequential / EM
  # ==========================================================

  speedup=$(
    awk \
      -v seq="$sequential_time" \
      -v em="$em_time" \
      'BEGIN {

          if (em > 0) {

              printf "%.3f", seq / em

          } else {

              printf "0.000"

          }
      }'
  )


  # ==========================================================
  # SM Utilization
  # ==========================================================

  sm_utilization=$(
    measure_sm_utilization "$stg"
  )


  # ==========================================================
  # Current result
  # ==========================================================

  if [ "$sm_utilization" = "N/A" ]; then

    printf "%-35s %22s %20s %14sx %20s\n" \
      "$name" \
      "$sequential_time" \
      "$em_time" \
      "$speedup" \
      "N/A"

  else

    printf "%-35s %22s %20s %14sx %19s%%\n" \
      "$name" \
      "$sequential_time" \
      "$em_time" \
      "$speedup" \
      "$sm_utilization"

  fi


  # ==========================================================
  # Save result for FINAL SUMMARY
  # ==========================================================

  RESULT_NAMES+=("$name")
  RESULT_SEQ+=("$sequential_time")
  RESULT_EM+=("$em_time")
  RESULT_SPEEDUP+=("$speedup")
  RESULT_SM+=("$sm_utilization")

done


# ============================================================
# FINAL SUMMARY
# ============================================================

echo
echo
echo "################################################################################################################################"
echo "# FINAL SUMMARY"
echo "################################################################################################################################"
echo


printf "%-35s %22s %20s %15s %20s\n" \
  "STG" \
  "Sequential Time [ms]" \
  "EM Time [ms]" \
  "Speedup [x]" \
  "SM Utilization [%]"


printf "%-35s %22s %20s %15s %20s\n" \
  "-----------------------------------" \
  "----------------------" \
  "--------------------" \
  "---------------" \
  "--------------------"


for ((i=0; i<${#RESULT_NAMES[@]}; i++)); do


  if [ "${RESULT_SM[$i]}" = "N/A" ]; then

    printf "%-35s %22s %20s %14sx %20s\n" \
      "${RESULT_NAMES[$i]}" \
      "${RESULT_SEQ[$i]}" \
      "${RESULT_EM[$i]}" \
      "${RESULT_SPEEDUP[$i]}" \
      "N/A"

  else

    printf "%-35s %22s %20s %14sx %19s%%\n" \
      "${RESULT_NAMES[$i]}" \
      "${RESULT_SEQ[$i]}" \
      "${RESULT_EM[$i]}" \
      "${RESULT_SPEEDUP[$i]}" \
      "${RESULT_SM[$i]}"

  fi

done


# ============================================================
# Information
# ============================================================

echo
echo "################################################################################################################################"
echo

echo "Execution Time:"
echo "  Runs    : $RUNS"
echo "  Warmup  : $WARMUP_RUNS"
echo "  Average : $((RUNS - WARMUP_RUNS)) runs"

echo

echo "Speedup:"
echo "  Sequential Time / EM Time"

echo

echo "SM Utilization:"
echo "  Average SMs Active [%]"
echo "  Initial sequential active region excluded"
echo "  Zero samples during EM execution included"

echo

echo "Nsight Systems Reports:"
echo "  $NSYS_RESULT_DIR"

echo
echo "Open example:"
echo "  nsys-ui $NSYS_RESULT_DIR/sample_fully_parallel_em.nsys-rep"

echo