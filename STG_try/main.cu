#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int kTaskCount = 4;
constexpr int kElementCount = 1 << 24;
constexpr int kIterations = 2000;

#define CUDA_CHECK(expression)                                                \
  do {                                                                        \
    const cudaError_t error = (expression);                                   \
    if (error != cudaSuccess) {                                               \
      throw std::runtime_error(                                               \
          std::string(cudaGetErrorName(error)) + ": " +                      \
          cudaGetErrorString(error) + " (" + __FILE__ + ":" +               \
          std::to_string(__LINE__) + ")");                                   \
    }                                                                         \
  } while (false)

struct GreenContextResources {
  std::array<cudaExecutionContext_t, kTaskCount> contexts{};
  std::array<cudaDevResourceDesc_t, kTaskCount> descriptors{};
  std::array<cudaStream_t, kTaskCount> streams{};
};

int parse_positive_int(const char* text, const std::string& name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);

  if (text == end || *end != '\0' || value <= 0 || value > INT_MAX) {
    throw std::invalid_argument(name + " must be a positive integer");
  }

  return static_cast<int>(value);
}

__global__ void task_kernel(float* output, int element_count, int iterations) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }

  float value = static_cast<float>(index & 255) * 0.001f;
#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
    value = value * 1.000001f + 0.00001f;
    value = value * 0.999999f + 0.00002f;
  }
  output[index] = value;
}

void destroy_resources(GreenContextResources& resources) noexcept {
  for (cudaStream_t& stream : resources.streams) {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }

  for (cudaExecutionContext_t& context : resources.contexts) {
    if (context != nullptr) {
      cudaExecutionCtxDestroy(context);
      context = nullptr;
    }
  }
}

GreenContextResources create_green_contexts(
    const std::array<int, kTaskCount>& requested_sm_counts,
    int device_index) {
  CUDA_CHECK(cudaSetDevice(device_index));

  cudaDevResource sm_resource{};
  CUDA_CHECK(cudaDeviceGetDevResource(
      device_index, &sm_resource, cudaDevResourceTypeSm));

  const int available_sm = static_cast<int>(sm_resource.sm.smCount);
  const int alignment =
      static_cast<int>(sm_resource.sm.smCoscheduledAlignment);
  const int allocation_unit = std::max(2, alignment);
  const int raw_minimum_sm =
      std::max(2, static_cast<int>(sm_resource.sm.minSmPartitionSize));
  const int minimum_sm =
      ((raw_minimum_sm + allocation_unit - 1) / allocation_unit) *
      allocation_unit;

  long long requested_total = 0;
  for (int task = 0; task < kTaskCount; ++task) {
    const int requested = requested_sm_counts[task];
    if (requested < minimum_sm) {
      throw std::invalid_argument(
          "task " + std::to_string(task) + " requests " +
          std::to_string(requested) + " SMs, but the minimum is " +
          std::to_string(minimum_sm));
    }
    if (requested % allocation_unit != 0) {
      throw std::invalid_argument(
          "task " + std::to_string(task) + " SM count must be a multiple of " +
          std::to_string(allocation_unit));
    }
    requested_total += requested;
  }

  if (requested_total > available_sm) {
    throw std::invalid_argument(
        "the four tasks request " + std::to_string(requested_total) +
        " SMs, but only " + std::to_string(available_sm) + " are available");
  }

  std::array<cudaDevResource, kTaskCount> split_resources{};
  std::array<cudaDevSmResourceGroupParams, kTaskCount> group_params{};
  for (int task = 0; task < kTaskCount; ++task) {
    group_params[task].smCount =
        static_cast<unsigned int>(requested_sm_counts[task]);
    group_params[task].coscheduledSmCount = 0;
    group_params[task].preferredCoscheduledSmCount = 0;
    group_params[task].flags = 0;
  }

  CUDA_CHECK(cudaDevSmResourceSplit(
      split_resources.data(), kTaskCount, &sm_resource, nullptr, 0,
      group_params.data()));

  GreenContextResources resources;
  try {
    for (int task = 0; task < kTaskCount; ++task) {
      CUDA_CHECK(cudaDevResourceGenerateDesc(
          &resources.descriptors[task], &split_resources[task], 1));
      CUDA_CHECK(cudaGreenCtxCreate(
          &resources.contexts[task], resources.descriptors[task],
          device_index, 0));
      CUDA_CHECK(cudaExecutionCtxStreamCreate(
          &resources.streams[task], resources.contexts[task],
          cudaStreamNonBlocking, 0));
    }
  } catch (...) {
    destroy_resources(resources);
    throw;
  }

  std::cout << "available SMs: " << available_sm
            << ", minimum partition: " << minimum_sm
            << ", allocation unit: " << allocation_unit << '\n';
  return resources;
}

void run_four_tasks(const std::array<int, kTaskCount>& sm_counts) {
  GreenContextResources resources = create_green_contexts(sm_counts, 0);
  std::array<float*, kTaskCount> device_outputs{};
  std::array<cudaEvent_t, kTaskCount> start_events{};
  std::array<cudaEvent_t, kTaskCount> stop_events{};

  try {
    const std::size_t bytes =
        static_cast<std::size_t>(kElementCount) * sizeof(float);
    for (int task = 0; task < kTaskCount; ++task) {
      CUDA_CHECK(cudaMalloc(&device_outputs[task], bytes));
      CUDA_CHECK(cudaEventCreate(&start_events[task]));
      CUDA_CHECK(cudaEventCreate(&stop_events[task]));
    }

    const auto wall_start = std::chrono::steady_clock::now();
    constexpr int threads_per_block = 256;
    const int block_count =
        (kElementCount + threads_per_block - 1) / threads_per_block;

    // Each independent task is submitted to its own Green Context stream.
    for (int task = 0; task < kTaskCount; ++task) {
      CUDA_CHECK(cudaEventRecord(start_events[task], resources.streams[task]));
      task_kernel<<<block_count, threads_per_block, 0,
                    resources.streams[task]>>>(
          device_outputs[task], kElementCount, kIterations);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(stop_events[task], resources.streams[task]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto wall_stop = std::chrono::steady_clock::now();

    std::cout << "===== Four Green Context tasks =====\n";
    for (int task = 0; task < kTaskCount; ++task) {
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(
          &elapsed_ms, start_events[task], stop_events[task]));
      std::cout << "task " << task << ": " << sm_counts[task]
                << " SM, " << elapsed_ms << " ms\n";
    }

    const double wall_ms =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start)
            .count();
    std::cout << "total wall time: " << wall_ms << " ms\n";
  } catch (...) {
    for (cudaEvent_t event : start_events) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    for (cudaEvent_t event : stop_events) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    for (float* output : device_outputs) {
      if (output != nullptr) cudaFree(output);
    }
    destroy_resources(resources);
    throw;
  }

  for (cudaEvent_t event : start_events) CUDA_CHECK(cudaEventDestroy(event));
  for (cudaEvent_t event : stop_events) CUDA_CHECK(cudaEventDestroy(event));
  for (float* output : device_outputs) CUDA_CHECK(cudaFree(output));
  destroy_resources(resources);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != kTaskCount + 1) {
    std::cerr << "Usage: " << argv[0] << " SM1 SM2 SM3 SM4\n"
              << "Example: " << argv[0] << " 16 16 16 16\n";
    return EXIT_FAILURE;
  }

  try {
    std::array<int, kTaskCount> sm_counts{};
    for (int task = 0; task < kTaskCount; ++task) {
      sm_counts[task] = parse_positive_int(
          argv[task + 1], "SM" + std::to_string(task + 1));
    }
    run_four_tasks(sm_counts);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
