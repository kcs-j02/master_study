#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>

struct BenchResult {
  double total_ms = 0.0;

  double stg_analysis_ms = 0.0;
  double task_importance_ms = 0.0;
  double stream_placement_ms = 0.0;
  double sm_allocation_comparison_ms = 0.0;
  double green_context_execution_ms = 0.0;

  double gpu_submit_wait_ms = 0.0;
  double gpu_kernel_ms = 0.0;
};

using Clock = std::chrono::steady_clock;

inline double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

inline double milliseconds_to_seconds(double milliseconds) {
  return milliseconds / 1000.0;
}

// スコープを抜けたら自動で時間を加算する
class ScopedTimer {
private:
  double& target_ms;
  Clock::time_point start;

public:
  explicit ScopedTimer(double& target)
      : target_ms(target), start(Clock::now()) {}

  ~ScopedTimer() {
    auto end = Clock::now();
    target_ms += elapsed_ms(start, end);
  }
};

inline BenchResult add_result(const BenchResult& a, const BenchResult& b) {
  BenchResult r;

  r.total_ms                 = a.total_ms + b.total_ms;
  r.stg_analysis_ms          = a.stg_analysis_ms + b.stg_analysis_ms;
  r.task_importance_ms       = a.task_importance_ms + b.task_importance_ms;
  r.stream_placement_ms      = a.stream_placement_ms + b.stream_placement_ms;
  r.sm_allocation_comparison_ms =
      a.sm_allocation_comparison_ms + b.sm_allocation_comparison_ms;
  r.green_context_execution_ms =
      a.green_context_execution_ms + b.green_context_execution_ms;
  r.gpu_submit_wait_ms       = a.gpu_submit_wait_ms + b.gpu_submit_wait_ms;
  r.gpu_kernel_ms            = a.gpu_kernel_ms + b.gpu_kernel_ms;

  return r;
}

inline BenchResult div_result(const BenchResult& a, double x) {
  if (x == 0.0) {
    throw std::invalid_argument("division by zero in div_result");
  }

  BenchResult r;

  r.total_ms                 = a.total_ms / x;
  r.stg_analysis_ms          = a.stg_analysis_ms / x;
  r.task_importance_ms       = a.task_importance_ms / x;
  r.stream_placement_ms      = a.stream_placement_ms / x;
  r.sm_allocation_comparison_ms =
      a.sm_allocation_comparison_ms / x;
  r.green_context_execution_ms =
      a.green_context_execution_ms / x;
  r.gpu_submit_wait_ms       = a.gpu_submit_wait_ms / x;
  r.gpu_kernel_ms            = a.gpu_kernel_ms / x;

  return r;
}

inline void print_result(const BenchResult& r) {
  const auto original_flags = std::cout.flags();
  const auto original_precision = std::cout.precision();

  std::cout << std::fixed << std::setprecision(9);
  std::cout << "===== Proposed Method Stage Times [seconds] =====\n";
  std::cout << "stage_1_stg_analysis_seconds: "
            << milliseconds_to_seconds(r.stg_analysis_ms) << " s\n";
  std::cout << "stage_2_task_importance_seconds: "
            << milliseconds_to_seconds(r.task_importance_ms) << " s\n";
  std::cout << "stage_3_stream_placement_seconds: "
            << milliseconds_to_seconds(r.stream_placement_ms) << " s\n";
  std::cout << "stage_4_sm_allocation_comparison_seconds: "
            << milliseconds_to_seconds(r.sm_allocation_comparison_ms)
            << " s\n";
  std::cout << "stage_5_green_context_execution_seconds: "
            << milliseconds_to_seconds(r.green_context_execution_ms)
            << " s\n";
  std::cout << "=================================================\n";

  std::cout.flags(original_flags);
  std::cout.precision(original_precision);

  std::cout << "========== Bench Result ==========\n";
  std::cout << "total_ms:                  " << r.total_ms << " ms\n";
  std::cout << "stage 1 STG analysis:      " << r.stg_analysis_ms << " ms\n";
  std::cout << "stage 2 task importance:   " << r.task_importance_ms << " ms\n";
  std::cout << "stage 3 stream placement:  " << r.stream_placement_ms << " ms\n";
  std::cout << "stage 4 SM comparison:     " << r.sm_allocation_comparison_ms << " ms\n";
  std::cout << "stage 5 GC execution:      " << r.green_context_execution_ms << " ms\n";
  std::cout << "gpu_submit_wait_ms:        " << r.gpu_submit_wait_ms << " ms\n";
  std::cout << "gpu_kernel_ms:             " << r.gpu_kernel_ms << " ms\n";
  std::cout << "==================================\n";
}
