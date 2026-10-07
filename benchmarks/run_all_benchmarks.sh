#!/usr/bin/env bash
set -euo pipefail

readonly benchmark_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$benchmark_root/benchmark_evaluation/run_all.sh" "$@"
