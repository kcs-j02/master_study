#!/usr/bin/env bash
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly root_dir="$(cd -- "$script_dir/../.." && pwd)"
readonly output="${1:-$script_dir/strict_ablation}"
readonly taskflow_include="${TASKFLOW_INC:-/home/kobayashi/taskflow}"
readonly nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"

if [[ ! -x "$nvcc" ]]; then
  echo "nvcc not found: $nvcc" >&2
  exit 1
fi

"$nvcc" -O2 -std=c++20 \
  "$script_dir/strict_ablation.cu" \
  -I"$taskflow_include" \
  -I"$root_dir" \
  -o "$output"

echo "built: $output"
