#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "bench_timer.hpp"
#include "../stg_common.hpp"

using StgGraph = stg::StgGraph;
using KernelKind = stg::KernelKind;
using TaskSpec = stg::TaskSpec;

constexpr int kTaskParallelSmLimit = 64;
constexpr int kTaskElementCount = kTaskParallelSmLimit * 256;

StgGraph load_stg_without_comm(const std::string& path) {
  return stg::load_stg_without_comm(path);
}

std::vector<TaskSpec> make_task_specs_from_stg(const StgGraph& graph) {
  return stg::make_task_specs_from_stg_common<TaskSpec>(
      graph,
      [](TaskSpec& spec, const stg::StgTask& task, int threshold) {
        spec.parallel_sm_limit = kTaskParallelSmLimit;

        if (task.proc_time > threshold) {
          spec.kind = KernelKind::HEAVY;
          spec.work_units = stg::safe_work_units(task.proc_time, 200, 1000);
        } else {
          spec.kind = KernelKind::LIGHT;
          spec.work_units = stg::safe_work_units(task.proc_time, 80, 200);
        }
      });
}

void launch_task_kernel(
    const TaskSpec& task,
    float* dmem,
    int n,
    cudaStream_t stream) {
  stg::launch_task_kernel(task, dmem, n, stream);
}

std::vector<std::size_t> make_topological_order(
    const std::vector<TaskSpec>& tasks) {
  return stg::make_topological_order(tasks);
}

BenchResult run_sequential_cuda(const std::vector<TaskSpec>& tasks) {
  BenchResult bench{};

  const auto total_start = Clock::now();
  const auto levelization_start = Clock::now();

  const std::vector<std::size_t> order =
      make_topological_order(tasks);

  const auto levelization_end = Clock::now();

  constexpr int N = kTaskElementCount;

  float* dmem = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;

  const auto resource_start = Clock::now();

  CUDA_CHECK(cudaMalloc(
      &dmem,
      static_cast<std::size_t>(N) * sizeof(float)));

  CUDA_CHECK(cudaMemset(
      dmem,
      0,
      static_cast<std::size_t>(N) * sizeof(float)));

  CUDA_CHECK(cudaStreamCreateWithFlags(
      &stream,
      cudaStreamNonBlocking));

  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  const auto resource_end = Clock::now();

  const auto submit_start = Clock::now();

  CUDA_CHECK(cudaEventRecord(start_event, stream));

  for (const std::size_t task_index : order) {
    const TaskSpec& task = tasks.at(task_index);

    launch_task_kernel(
        task,
        dmem,
        N,
        stream);
  }

  CUDA_CHECK(cudaEventRecord(stop_event, stream));
  CUDA_CHECK(cudaEventSynchronize(stop_event));

  const auto submit_end = Clock::now();

  float gpu_kernel_ms = 0.0f;

  CUDA_CHECK(cudaEventElapsedTime(
      &gpu_kernel_ms,
      start_event,
      stop_event));

  bench.gpu_kernel_ms =
      static_cast<double>(gpu_kernel_ms);

  bench.gpu_submit_wait_ms =
      elapsed_ms(submit_start, submit_end);

  bench.task_DFG_construction_ms = 0.0;
  bench.task_levelization_ms =
      elapsed_ms(levelization_start, levelization_end);
  bench.task_assignment_ms = 0.0;
  bench.resource_allocation_ms =
      elapsed_ms(resource_start, resource_end);

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dmem));

  const auto total_end = Clock::now();

  bench.total_ms =
      elapsed_ms(total_start, total_end);

  return bench;
}

void print_stg_summary(const std::vector<TaskSpec>& tasks) {
  stg::print_stg_summary(tasks);

  std::cout
      << "task_grid      : "
      << kTaskParallelSmLimit
      << " blocks\n";

  std::cout
      << "task_SM_limit  : "
      << kTaskParallelSmLimit
      << " SM\n";
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr
        << "Usage: "
        << argv[0]
        << " input.stg\n";

    return EXIT_FAILURE;
  }

  try {
    const StgGraph graph =
        load_stg_without_comm(argv[1]);

    const std::vector<TaskSpec> tasks =
        make_task_specs_from_stg(graph);

    print_stg_summary(tasks);

    const BenchResult result =
        run_sequential_cuda(tasks);

    std::cout << "mode: sequential\n";
    print_result(result);
  } catch (const std::exception& error) {
    std::cerr
        << "exception: "
        << error.what()
        << "\n";

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
