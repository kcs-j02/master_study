#!/usr/bin/env bash
set -euo pipefail
readonly here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$here/run_single_method.sh" existing_method
