#!/usr/bin/env bash
set -euo pipefail

readonly benchmark_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly master_root="$(cd -- "$benchmark_root/.." && pwd)"
readonly taskflow_include="${TASKFLOW_INC:-/home/kobayashi/taskflow}"
readonly nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"

if [[ ! -x "$nvcc" ]]; then
  echo "nvcc not found: $nvcc" >&2
  exit 1
fi

if [[ ! -d "$taskflow_include" ]]; then
  echo "Taskflow include directory not found: $taskflow_include" >&2
  exit 1
fi

build_method() {
  local method="$1"
  local source="$master_root/$method/main.cu"
  local executable="$master_root/$method/main"

  if [[ ! -f "$source" ]]; then
    echo "method source not found: $source" >&2
    exit 1
  fi

  echo "building $method..." >&2
  "$nvcc" -O2 -std=c++20 "$source" \
    -I"$taskflow_include" \
    -I"$master_root" \
    -o "$executable"
}

for method in \
  sequential_method \
  existing_method \
  existing_method_green_context \
  proposed_method
do
  build_method "$method"
done

exec "$benchmark_root/benchmark_evaluation/run_all.sh" "$@"
