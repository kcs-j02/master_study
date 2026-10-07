#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../proposed_method/05_green_context_execution.cuh"

namespace {

struct Measurement {
  int sm_count = 0;
  double median_ms = 0.0;
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status)
    );
  }
}

double median(std::vector<float> values) {
  if (values.empty()) {
    throw std::invalid_argument("median requires at least one value");
  }

  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 != 0) {
    return static_cast<double>(values.at(middle));
  }

  return (
      static_cast<double>(values.at(middle - 1)) +
      static_cast<double>(values.at(middle))
  ) / 2.0;
}

double measure_task(
    const TaskSpec& task,
    int sm_count,
    float* device_memory,
    int warmup_runs,
    int measured_runs
) {
  const bool use_full_gpu =
      sm_count == kSchedulingReferenceSmCount;
  const std::vector<int> allocation =
      use_full_gpu
          ? std::vector<int>{kSchedulingReferenceSmCount}
          : std::vector<int>{
                kSchedulingReferenceSmCount - sm_count,
                sm_count
            };

  RuntimeResources resources =
      create_green_context_resources(allocation);
  const int stream_id = use_full_gpu ? 0 : 1;
  const cudaStream_t stream =
      resources.streams.at(static_cast<std::size_t>(stream_id));

  for (int run = 0; run < warmup_runs; ++run) {
    stage5_detail::launch_configured_task_kernel(
        task,
        device_memory,
        kTaskElementCount,
        stream,
        sm_count
    );
  }
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  std::vector<float> elapsed_times;
  elapsed_times.reserve(static_cast<std::size_t>(measured_runs));

  for (int run = 0; run < measured_runs; ++run) {
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    stage5_detail::launch_configured_task_kernel(
        task,
        device_memory,
        kTaskElementCount,
        stream,
        sm_count
    );
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&elapsed_ms, start, stop),
        "cudaEventElapsedTime"
    );
    elapsed_times.push_back(elapsed_ms);
  }

  check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  destroy_runtime_resources(resources);

  return median(std::move(elapsed_times));
}

std::vector<Measurement> benchmark_kind(
    const std::string& label,
    KernelKind kind,
    int work_units,
    const std::vector<int>& sm_counts,
    float* device_memory
) {
  constexpr int warmup_runs = 5;
  constexpr int measured_runs = 21;

  TaskSpec task;
  task.id = 0;
  task.proc_time = 1;
  task.kind = kind;
  task.work_units = work_units;
  task.parallel_sm_limit = kTaskParallelSmLimit;

  std::vector<Measurement> measurements;
  measurements.reserve(sm_counts.size());

  std::cout << "\n[" << label << "] work_units=" << work_units << '\n';
  for (const int sm_count : sm_counts) {
    const double elapsed_ms = measure_task(
        task,
        sm_count,
        device_memory,
        warmup_runs,
        measured_runs
    );
    measurements.push_back(Measurement{sm_count, elapsed_ms});
    std::cout << "measured SM=" << std::setw(3) << sm_count
              << " median_ms=" << std::fixed << std::setprecision(6)
              << elapsed_ms << '\n';
  }

  const double reference_ms = measurements.back().median_ms;
  std::cout
      << "\nkind,sm_count,median_ms,measured_slowdown,linear_114_over_sm,"
         "measured_over_linear\n";
  for (const auto& measurement : measurements) {
    const double measured_slowdown =
        measurement.median_ms / reference_ms;
    const double linear_slowdown =
        static_cast<double>(kSchedulingReferenceSmCount) /
        static_cast<double>(measurement.sm_count);

    std::cout
        << label << ','
        << measurement.sm_count << ','
        << std::fixed << std::setprecision(6)
        << measurement.median_ms << ','
        << measured_slowdown << ','
        << linear_slowdown << ','
        << measured_slowdown / linear_slowdown << '\n';
  }

  return measurements;
}

}  // namespace

int main() {
  try {
    const SmPartitionInfo partition = query_sm_partition_info();
    validate_fixed_sm_table_compatibility(partition);

    std::vector<int> sm_counts;
    for (int sm_count = partition.min_group_sm;
         sm_count < partition.available_sm;
         sm_count += partition.unit_sm) {
      sm_counts.push_back(sm_count);
    }
    sm_counts.push_back(partition.available_sm);

    float* device_memory = nullptr;
    check_cuda(cudaMalloc(
        &device_memory,
        static_cast<std::size_t>(kTaskElementCount) * sizeof(float)
    ), "cudaMalloc");
    check_cuda(cudaMemset(
        device_memory,
        0,
        static_cast<std::size_t>(kTaskElementCount) * sizeof(float)
    ), "cudaMemset");

    std::cout << "GPU available SM=" << partition.available_sm
              << ", GC unit=" << partition.unit_sm
              << ", GC minimum=" << partition.min_group_sm << '\n';
    std::cout << "Each value is the median of 21 CUDA-event measurements.\n";

    benchmark_kind(
        "LIGHT",
        KernelKind::LIGHT,
        650 * 80,
        sm_counts,
        device_memory
    );
    benchmark_kind(
        "HEAVY",
        KernelKind::HEAVY,
        1000 * 200,
        sm_counts,
        device_memory
    );

    check_cuda(cudaFree(device_memory), "cudaFree");
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
