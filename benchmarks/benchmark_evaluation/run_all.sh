#!/usr/bin/env bash
set -euo pipefail
readonly project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$project_dir/scripts/run_all_four_methods.sh"
