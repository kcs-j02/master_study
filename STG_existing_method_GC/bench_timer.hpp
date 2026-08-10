#pragma once

#include <chrono>
#include <iostream>
#include <stdexcept>

struct BenchResult {
  double total_ms = 0.0;

  double task_DFG_construction_ms = 0.0;
  double task_levelization_ms = 0.0;
  double task_assignment_ms = 0.0;
  double resource_allocation_ms = 0.0;

  double gpu_submit_wait_ms = 0.0;
  double gpu_kernel_ms = 0.0;
};

using Clock = std::chrono::steady_clock;

inline double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
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
  r.task_DFG_construction_ms = a.task_DFG_construction_ms + b.task_DFG_construction_ms;
  r.task_levelization_ms     = a.task_levelization_ms + b.task_levelization_ms;
  r.task_assignment_ms       = a.task_assignment_ms + b.task_assignment_ms;
  r.resource_allocation_ms   = a.resource_allocation_ms + b.resource_allocation_ms;
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
  r.task_DFG_construction_ms = a.task_DFG_construction_ms / x;
  r.task_levelization_ms     = a.task_levelization_ms / x;
  r.task_assignment_ms       = a.task_assignment_ms / x;
  r.resource_allocation_ms   = a.resource_allocation_ms / x;
  r.gpu_submit_wait_ms       = a.gpu_submit_wait_ms / x;
  r.gpu_kernel_ms            = a.gpu_kernel_ms / x;

  return r;
}

inline void print_result(const BenchResult& r) {
  std::cout << "========== Bench Result ==========\n";
  std::cout << "total_ms:                  " << r.total_ms << " ms\n";
  std::cout << "task DFG construction:     " << r.task_DFG_construction_ms << " ms\n";
  std::cout << "task levelization:         " << r.task_levelization_ms << " ms\n";
  std::cout << "task assignment:           " << r.task_assignment_ms << " ms\n";
  std::cout << "resource allocation:       " << r.resource_allocation_ms << " ms\n";
  std::cout << "gpu_submit_wait_ms:        " << r.gpu_submit_wait_ms << " ms\n";
  std::cout << "gpu_kernel_ms:             " << r.gpu_kernel_ms << " ms\n";
  std::cout << "==================================\n";
}