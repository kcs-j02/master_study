#!/usr/bin/env bash

# Single source of truth for every executable benchmark input.
ALL_BENCHMARKS=(
  vector_square
  black_scholes
  machine_learning
  hits
  image_processing
  deep_learning
  synthetic_fully_parallel
  synthetic_mixed_chain_parallel
  synthetic_multiple_long_branches
  synthetic_random_dag
  synthetic_sequential_chain
)

is_known_benchmark() {
  local requested="$1"
  local catalog_entry
  for catalog_entry in "${ALL_BENCHMARKS[@]}"; do
    if [[ "$catalog_entry" == "$requested" ]]; then
      return 0
    fi
  done
  return 1
}

benchmark_group() {
  case "$1" in
    synthetic_*) printf '%s\n' STG ;;
    *) printf '%s\n' KESCO ;;
  esac
}
