#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

constexpr int TASKS = 4;
constexpr int N = 1 << 24;

void check_cuda(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at "
              << what << "\n";
    std::exit(EXIT_FAILURE);
  }
}

__global__ void kernel(float* data, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float x = i;
  for (int j = 0; j < iters; ++j) {
    x = x * 1.000001f + 0.00001f;
  }
  data[i] = x;
}

struct BenchmarkCaseResult {
  std::array<float, TASKS> parallel_ms{};
  std::array<float, TASKS> sequential_ms{};
  double parallel_total_ms = 0.0;
  double sequential_total_ms = 0.0;
};

struct SingleTaskCaseResult {
  float kernel_ms = 0.0f;
  double total_ms = 0.0;
};

SingleTaskCaseResult run_single_task_case(
    int kernel_iters,
    int sm_count,
    cudaDevResource all_sms) {
  SingleTaskCaseResult result{};

  cudaDevResource split_sms{};
  cudaDevSmResourceGroupParams param{};
  param.smCount = sm_count;

  check_cuda(cudaDevSmResourceSplit(
                 &split_sms, 1, &all_sms, nullptr, 0, &param),
             "cudaDevSmResourceSplit(single)");

  cudaDevResourceDesc_t desc{};
  cudaExecutionContext_t context{};
  cudaStream_t stream{};
  cudaEvent_t start{};
  cudaEvent_t stop{};
  float* data = nullptr;

  check_cuda(cudaDevResourceGenerateDesc(&desc, &split_sms, 1),
             "cudaDevResourceGenerateDesc(single)");
  check_cuda(cudaGreenCtxCreate(&context, desc, 0, 0),
             "cudaGreenCtxCreate(single)");
  check_cuda(cudaExecutionCtxStreamCreate(
                 &stream, context, cudaStreamNonBlocking, 0),
             "cudaExecutionCtxStreamCreate(single)");
  check_cuda(cudaMalloc(&data, N * sizeof(float)), "cudaMalloc(single)");
  check_cuda(cudaEventCreate(&start), "cudaEventCreate(start,single)");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop,single)");

  auto total_start = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start,single)");
  kernel<<<(N + 255) / 256, 256, 0, stream>>>(data, kernel_iters);
  check_cuda(cudaGetLastError(), "cudaGetLastError(single)");
  check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop,single)");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(single)");
  auto total_stop = std::chrono::steady_clock::now();

  check_cuda(cudaEventElapsedTime(&result.kernel_ms, start, stop),
             "cudaEventElapsedTime(single)");
  result.total_ms =
      std::chrono::duration<double, std::milli>(total_stop - total_start)
          .count();

  check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start,single)");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop,single)");
  check_cuda(cudaFree(data), "cudaFree(single)");
  check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(single)");
  check_cuda(cudaExecutionCtxDestroy(context),
             "cudaExecutionCtxDestroy(single)");

  return result;
}

BenchmarkCaseResult run_benchmark_case(
    int kernel_iters,
    const std::array<int, TASKS>& sm_count,
    cudaDevResource all_sms) {
  BenchmarkCaseResult result{};

  std::array<cudaDevResource, TASKS> split_sms{};
  std::array<cudaDevSmResourceGroupParams, TASKS> params{};
  for (int i = 0; i < TASKS; ++i) {
    params[i].smCount = sm_count[i];
  }

  check_cuda(
      cudaDevSmResourceSplit(
          split_sms.data(), TASKS, &all_sms, nullptr, 0, params.data()),
      "cudaDevSmResourceSplit");

  std::array<cudaDevResourceDesc_t, TASKS> desc{};
  std::array<cudaExecutionContext_t, TASKS> context{};
  std::array<cudaStream_t, TASKS> stream{};
  std::array<cudaEvent_t, TASKS> start{};
  std::array<cudaEvent_t, TASKS> stop{};
  std::array<float*, TASKS> data{};

  for (int i = 0; i < TASKS; ++i) {
    check_cuda(
        cudaDevResourceGenerateDesc(&desc[i], &split_sms[i], 1),
        "cudaDevResourceGenerateDesc");
    check_cuda(cudaGreenCtxCreate(&context[i], desc[i], 0, 0),
               "cudaGreenCtxCreate");
    check_cuda(cudaExecutionCtxStreamCreate(
                   &stream[i], context[i], cudaStreamNonBlocking, 0),
               "cudaExecutionCtxStreamCreate");
    check_cuda(cudaMalloc(&data[i], N * sizeof(float)), "cudaMalloc");
    check_cuda(cudaEventCreate(&start[i]), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop[i]), "cudaEventCreate(stop)");
  }

  auto total_start = std::chrono::steady_clock::now();
  for (int i = 0; i < TASKS; ++i) {
    check_cuda(cudaEventRecord(start[i], stream[i]), "cudaEventRecord(start)");
    kernel<<<(N + 255) / 256, 256, 0, stream[i]>>>(data[i], kernel_iters);
    check_cuda(cudaGetLastError(), "cudaGetLastError");
    check_cuda(cudaEventRecord(stop[i], stream[i]), "cudaEventRecord(stop)");
  }

  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  auto total_stop = std::chrono::steady_clock::now();

  for (int i = 0; i < TASKS; ++i) {
    check_cuda(cudaEventElapsedTime(&result.parallel_ms[i], start[i], stop[i]),
               "cudaEventElapsedTime(parallel)");
    check_cuda(cudaEventDestroy(start[i]), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop[i]), "cudaEventDestroy(stop)");
    check_cuda(cudaFree(data[i]), "cudaFree(data)");
    check_cuda(cudaStreamDestroy(stream[i]), "cudaStreamDestroy(stream)");
    check_cuda(cudaExecutionCtxDestroy(context[i]), "cudaExecutionCtxDestroy");
  }

  result.parallel_total_ms =
      std::chrono::duration<double, std::milli>(total_stop - total_start)
          .count();

  cudaDevResource sequential_sms{};
  cudaDevSmResourceGroupParams sequential_param{};
  sequential_param.smCount = 64;
  check_cuda(cudaDevSmResourceSplit(
                 &sequential_sms, 1, &all_sms, nullptr, 0, &sequential_param),
             "cudaDevSmResourceSplit(sequential)");

  cudaDevResourceDesc_t sequential_desc{};
  cudaExecutionContext_t sequential_context{};
  cudaStream_t sequential_stream{};
  check_cuda(cudaDevResourceGenerateDesc(
                 &sequential_desc, &sequential_sms, 1),
             "cudaDevResourceGenerateDesc(sequential)");
  check_cuda(cudaGreenCtxCreate(&sequential_context, sequential_desc, 0, 0),
             "cudaGreenCtxCreate(sequential)");
  check_cuda(cudaExecutionCtxStreamCreate(
                 &sequential_stream, sequential_context,
                 cudaStreamNonBlocking, 0),
             "cudaExecutionCtxStreamCreate(sequential)");

  for (int i = 0; i < TASKS; ++i) {
    check_cuda(cudaMalloc(&data[i], N * sizeof(float)), "cudaMalloc(sequential)");
    check_cuda(cudaEventCreate(&start[i]), "cudaEventCreate(start,sequential)");
    check_cuda(cudaEventCreate(&stop[i]), "cudaEventCreate(stop,sequential)");
  }

  auto sequential_total_start = std::chrono::steady_clock::now();
  for (int i = 0; i < TASKS; ++i) {
    check_cuda(cudaEventRecord(start[i], sequential_stream),
               "cudaEventRecord(start,sequential)");
    kernel<<<(N + 255) / 256, 256, 0, sequential_stream>>>(data[i], kernel_iters);
    check_cuda(cudaGetLastError(), "cudaGetLastError(sequential)");
    check_cuda(cudaEventRecord(stop[i], sequential_stream),
               "cudaEventRecord(stop,sequential)");
  }
  check_cuda(cudaStreamSynchronize(sequential_stream), "cudaStreamSynchronize");
  auto sequential_total_stop = std::chrono::steady_clock::now();

  for (int i = 0; i < TASKS; ++i) {
    check_cuda(cudaEventElapsedTime(&result.sequential_ms[i], start[i], stop[i]),
               "cudaEventElapsedTime(sequential)");
    check_cuda(cudaEventDestroy(start[i]), "cudaEventDestroy(start,sequential)");
    check_cuda(cudaEventDestroy(stop[i]), "cudaEventDestroy(stop,sequential)");
    check_cuda(cudaFree(data[i]), "cudaFree(data,sequential)");
  }

  result.sequential_total_ms =
      std::chrono::duration<double, std::milli>(
          sequential_total_stop - sequential_total_start)
          .count();

  check_cuda(cudaStreamDestroy(sequential_stream), "cudaStreamDestroy(sequential)");
  check_cuda(cudaExecutionCtxDestroy(sequential_context),
             "cudaExecutionCtxDestroy(sequential)");

  return result;
}

int main(int argc, char** argv) {
  if (argc != TASKS + 1) {
    std::cerr << "Usage: " << argv[0] << " SM1 SM2 SM3 SM4\n";
    return 1;
  }

  std::array<int, TASKS> sm_count{};
  for (int i = 0; i < TASKS; ++i) {
    sm_count[i] = std::atoi(argv[i + 1]);
    if (sm_count[i] <= 0) {
      std::cerr << "SM count must be positive\n";
      return 1;
    }
  }

  cudaDevResource all_sms{};
  check_cuda(cudaDeviceGetDevResource(0, &all_sms, cudaDevResourceTypeSm),
             "cudaDeviceGetDevResource");

  const auto short_result = run_benchmark_case(2000, sm_count, all_sms);
  const auto medium_result = run_benchmark_case(200000, sm_count, all_sms);
  const auto long_result = run_benchmark_case(1000000, sm_count, all_sms);

  const std::array<int, 3> single_task_sms = {16, 32, 64};
  std::array<SingleTaskCaseResult, 3> single_task_results{};
  for (int i = 0; i < 3; ++i) {
    single_task_results[i] = run_single_task_case(
        1000000, single_task_sms[i], all_sms);
  }

  auto print_table = [&](const char* title, const BenchmarkCaseResult& result) {
    std::cout << "\n" << title << "\n";
    std::cout << std::left << std::setw(8) << "task"
              << "| " << std::setw(24) << "4 Green Contexts (parallel)"
              << "| " << std::setw(24) << "64 SM stream (sequential)"
              << "\n";
    std::cout << std::string(8 + 1 + 24 + 1 + 24 + 1, '-') << "\n";

    for (int i = 0; i < TASKS; ++i) {
      std::ostringstream parallel_ss;
      parallel_ss << sm_count[i] << " SM, " << result.parallel_ms[i] << " ms";
      std::ostringstream sequential_ss;
      sequential_ss << "64 SM, " << result.sequential_ms[i] << " ms";

      std::cout << std::left << std::setw(8) << i
                << "| " << std::setw(24) << parallel_ss.str()
                << "| " << std::setw(24) << sequential_ss.str()
                << "\n";
    }

    std::ostringstream parallel_total_ss;
    parallel_total_ss << "total, " << result.parallel_total_ms << " ms";
    std::ostringstream sequential_total_ss;
    sequential_total_ss << "total, " << result.sequential_total_ms << " ms";

    std::cout << std::left << std::setw(8) << "total"
              << "| " << std::setw(24) << parallel_total_ss.str()
              << "| " << std::setw(24) << sequential_total_ss.str()
              << "\n";
  };

  print_table("[short workload: 2000 iterations]", short_result);
  print_table("[medium workload: 200000 iterations]", medium_result);
  print_table("[long workload: 1000000 iterations]", long_result);

  std::cout << "\n[one task SM scaling: 1000000 iterations]\n";
  std::cout << std::left << std::setw(8) << "SM"
            << "| " << std::setw(24) << "kernel time (ms)"
            << "\n";
  std::cout << std::string(8 + 1 + 24 + 1, '-') << "\n";
  for (int i = 0; i < 3; ++i) {
    std::cout << std::left << std::setw(8) << single_task_sms[i]
              << "| " << std::setw(24) << single_task_results[i].kernel_ms
              << "\n";
  }
  std::cout << "\n";

  return 0;
}
