#pragma once

#include <cuda_runtime.h>
#include <taskflow/taskflow.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "01_stg_analysis.hpp"
#include "03_stream_assignment.hpp"
#include "04_sm_allocation_comparison.hpp"

/*
 * Stage 5: Green Contextを用いた実行。
 *
 * Stage 1--4で確定したタスク配置とSM配分だけを受け取り、CUDA資源の
 * 構築、依存イベントの設定、kernel投入、完了待ち、資源解放を行う。
 */

struct GreenContextExecutionOptions {
  bool disable_green_context = false;
  int background_chunk_count = 1;
  int scheduling_reference_sm = 114;
};

struct GreenContextExecutionResult {
  double resource_setup_ms = 0.0;
  double gpu_submit_wait_ms = 0.0;
  double gpu_kernel_ms = 0.0;
};

#define GC_CUDA_CHECK(expression)                                             \
  do {                                                                        \
    const cudaError_t gc_error__ = (expression);                              \
    if (gc_error__ != cudaSuccess) {                                          \
      throw std::runtime_error(                                               \
          std::string("CUDA error: ") + cudaGetErrorString(gc_error__) +     \
          " at " + __FILE__ + ":" + std::to_string(__LINE__)               \
      );                                                                      \
    }                                                                         \
  } while (0)

struct RuntimeResources {
  std::vector<cudaStream_t> streams;
  std::vector<cudaExecutionContext_t> green_contexts;
  std::vector<cudaDevResourceDesc_t> resource_descriptors;
  std::vector<int> sm_count_each_stream;
  std::vector<long long> priority_each_stream;
  int normal_stream_id = 0;
  std::vector<int> green_context_stream_ids;
};

inline cudaStream_t get_stream_by_id(
    const RuntimeResources& resources,
    int stream_id
) {
  if (stream_id < 0 ||
      stream_id >= static_cast<int>(resources.streams.size())) {
    throw std::runtime_error(
        "invalid stream id: " + std::to_string(stream_id)
    );
  }

  const cudaStream_t stream = resources.streams.at(
      static_cast<std::size_t>(stream_id)
  );
  if (stream == nullptr) {
    throw std::runtime_error(
        "stream is not available: " + std::to_string(stream_id)
    );
  }

  return stream;
}

inline bool is_green_context_stream_id(
    const RuntimeResources& resources,
    int stream_id
) {
  return std::find(
      resources.green_context_stream_ids.begin(),
      resources.green_context_stream_ids.end(),
      stream_id
  ) != resources.green_context_stream_ids.end();
}

inline RuntimeResources create_normal_stream_resources(
    int stream_count,
    int scheduling_reference_sm
) {
  if (stream_count <= 0 || scheduling_reference_sm <= 0) {
    throw std::invalid_argument(
        "stream_count and scheduling_reference_sm must be positive"
    );
  }

  GC_CUDA_CHECK(cudaSetDevice(0));

  RuntimeResources resources;
  resources.streams.resize(
      static_cast<std::size_t>(stream_count),
      nullptr
  );
  resources.green_contexts.resize(
      static_cast<std::size_t>(stream_count),
      nullptr
  );
  resources.resource_descriptors.resize(
      static_cast<std::size_t>(stream_count),
      nullptr
  );
  resources.sm_count_each_stream.resize(
      static_cast<std::size_t>(stream_count),
      scheduling_reference_sm
  );
  resources.priority_each_stream.resize(
      static_cast<std::size_t>(stream_count),
      0
  );
  resources.normal_stream_id = -1;

  int least_priority = 0;
  int greatest_priority = 0;
  GC_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(
      &least_priority,
      &greatest_priority
  ));

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    const int priority =
        stream_id == 0 ? greatest_priority : least_priority;

    GC_CUDA_CHECK(cudaStreamCreateWithPriority(
        &resources.streams.at(static_cast<std::size_t>(stream_id)),
        cudaStreamNonBlocking,
        priority
    ));
    resources.priority_each_stream.at(
        static_cast<std::size_t>(stream_id)
    ) = priority;
  }

  std::cout
      << "[adaptive resource mode] shared priority streams, no GC\n"
      << "critical stream priority: " << greatest_priority << '\n'
      << "background priority     : " << least_priority << '\n';

  return resources;
}

inline RuntimeResources create_green_context_resources(
    const std::vector<int>& stream_sm_counts,
    int gpu_device_index = 0
) {
  if (stream_sm_counts.empty()) {
    throw std::invalid_argument("at least one stream is required");
  }

  GC_CUDA_CHECK(cudaSetDevice(gpu_device_index));
  const SmPartitionInfo partition_info =
      query_sm_partition_info(gpu_device_index);
  const int stream_count =
      static_cast<int>(stream_sm_counts.size());
  const int green_context_count = stream_count - 1;

  int requested_green_context_sm = 0;
  for (int stream_id = 1;
       stream_id < stream_count;
       ++stream_id) {
    const int sm_count = stream_sm_counts.at(
        static_cast<std::size_t>(stream_id)
    );

    if (sm_count < partition_info.min_group_sm) {
      throw std::invalid_argument(
          "GC SM count is smaller than minimum: stream " +
          std::to_string(stream_id)
      );
    }
    if (sm_count % partition_info.unit_sm != 0) {
      throw std::invalid_argument(
          "GC SM count is not aligned: stream " +
          std::to_string(stream_id)
      );
    }

    requested_green_context_sm += sm_count;
  }

  if (requested_green_context_sm >= partition_info.available_sm) {
    throw std::runtime_error(
        "Green Contexts must leave SMs for the normal stream"
    );
  }

  RuntimeResources resources;
  resources.streams.resize(
      static_cast<std::size_t>(stream_count),
      nullptr
  );
  resources.green_contexts.resize(
      static_cast<std::size_t>(stream_count),
      nullptr
  );
  resources.resource_descriptors.resize(
      static_cast<std::size_t>(stream_count),
      nullptr
  );
  resources.sm_count_each_stream = stream_sm_counts;
  resources.priority_each_stream.resize(
      static_cast<std::size_t>(stream_count),
      0
  );
  resources.normal_stream_id = 0;
  resources.green_context_stream_ids.reserve(
      static_cast<std::size_t>(green_context_count)
  );

  for (int stream_id = 1;
       stream_id < stream_count;
       ++stream_id) {
    resources.green_context_stream_ids.push_back(stream_id);
  }

  if (stream_count == 1) {
    if (stream_sm_counts.at(0) != partition_info.available_sm) {
      throw std::invalid_argument(
          "single normal stream must use all available SMs"
      );
    }

    GC_CUDA_CHECK(cudaStreamCreateWithFlags(
        &resources.streams.at(0),
        cudaStreamNonBlocking
    ));
    std::cout << "[normal stream setup] no Green Context\n";
    return resources;
  }

  cudaDevResource initial_sm_resource {};
  GC_CUDA_CHECK(cudaDeviceGetDevResource(
      gpu_device_index,
      &initial_sm_resource,
      cudaDevResourceTypeSm
  ));

  std::vector<cudaDevResource> split_sm_resources(
      static_cast<std::size_t>(green_context_count)
  );
  std::vector<cudaDevSmResourceGroupParams> group_parameters(
      static_cast<std::size_t>(green_context_count)
  );

  for (int green_context_index = 0;
       green_context_index < green_context_count;
       ++green_context_index) {
    const int stream_id = green_context_index + 1;
    auto& parameters = group_parameters.at(
        static_cast<std::size_t>(green_context_index)
    );
    parameters = {};
    parameters.smCount = static_cast<unsigned int>(
        stream_sm_counts.at(static_cast<std::size_t>(stream_id))
    );
    parameters.coscheduledSmCount = 2;
    parameters.preferredCoscheduledSmCount = 2;
    parameters.flags = 0;
  }

  GC_CUDA_CHECK(cudaDeviceSynchronize());
  const auto setup_start = std::chrono::steady_clock::now();

  const cudaError_t split_error = cudaDevSmResourceSplit(
      split_sm_resources.data(),
      static_cast<unsigned int>(green_context_count),
      &initial_sm_resource,
      nullptr,
      0,
      group_parameters.data()
  );

  if (split_error != cudaSuccess) {
    std::cerr
        << "cudaDevSmResourceSplit failed\n"
        << "available_sm=" << partition_info.available_sm
        << ", requested_gc_sm=" << requested_green_context_sm
        << ", gc_stream_count=" << green_context_count
        << '\n';
    GC_CUDA_CHECK(split_error);
  }

  for (int green_context_index = 0;
       green_context_index < green_context_count;
       ++green_context_index) {
    const int stream_id = green_context_index + 1;
    GC_CUDA_CHECK(cudaDevResourceGenerateDesc(
        &resources.resource_descriptors.at(
            static_cast<std::size_t>(stream_id)
        ),
        &split_sm_resources.at(
            static_cast<std::size_t>(green_context_index)
        ),
        1
    ));
  }

  for (int stream_id = 1;
       stream_id < stream_count;
       ++stream_id) {
    GC_CUDA_CHECK(cudaGreenCtxCreate(
        &resources.green_contexts.at(
            static_cast<std::size_t>(stream_id)
        ),
        resources.resource_descriptors.at(
            static_cast<std::size_t>(stream_id)
        ),
        gpu_device_index,
        0
    ));
  }

  for (int stream_id = 1;
       stream_id < stream_count;
       ++stream_id) {
    GC_CUDA_CHECK(cudaExecutionCtxStreamCreate(
        &resources.streams.at(static_cast<std::size_t>(stream_id)),
        resources.green_contexts.at(
            static_cast<std::size_t>(stream_id)
        ),
        cudaStreamNonBlocking,
        0
    ));
  }

  GC_CUDA_CHECK(cudaStreamCreateWithFlags(
      &resources.streams.at(0),
      cudaStreamNonBlocking
  ));

  const auto setup_end = std::chrono::steady_clock::now();
  const double setup_ms =
      std::chrono::duration<double, std::milli>(
          setup_end - setup_start
      ).count();
  std::cout << "[GC setup time] " << setup_ms << " ms\n";

  return resources;
}

inline void print_runtime_resource_state(
    const RuntimeResources& resources,
    const std::string& tag
) {
  std::cout
      << "\n===== Runtime Resource State: "
      << tag << " =====\n";

  for (int stream_id = 0;
       stream_id < static_cast<int>(resources.streams.size());
       ++stream_id) {
    const std::size_t index = static_cast<std::size_t>(stream_id);
    std::cout << "stream[" << stream_id << "] ";

    if (stream_id == resources.normal_stream_id) {
      std::cout << "[normal] ";
    }
    else if (is_green_context_stream_id(resources, stream_id)) {
      std::cout << "[GC] ";
    }
    else {
      std::cout << "[unused] ";
    }

    std::cout
        << "stream_ptr=" << resources.streams.at(index)
        << ", green_ctx=" << resources.green_contexts.at(index)
        << ", has_stream=" << (resources.streams.at(index) != nullptr)
        << ", has_gc=" <<
            (resources.green_contexts.at(index) != nullptr)
        << ", SM=" << resources.sm_count_each_stream.at(index)
        << '\n';
  }

  std::cout << "=============================================\n";
}

inline void release_green_context_stream(
    RuntimeResources& resources,
    int stream_id
) {
  if (!is_green_context_stream_id(resources, stream_id)) {
    return;
  }

  const std::size_t index = static_cast<std::size_t>(stream_id);
  if (resources.green_contexts.at(index) == nullptr) {
    return;
  }

  const int released_sm_count =
      resources.sm_count_each_stream.at(index);

  if (resources.streams.at(index) != nullptr) {
    GC_CUDA_CHECK(cudaStreamSynchronize(resources.streams.at(index)));
    GC_CUDA_CHECK(cudaStreamDestroy(resources.streams.at(index)));
    resources.streams.at(index) = nullptr;
  }

  GC_CUDA_CHECK(cudaExecutionCtxDestroy(
      resources.green_contexts.at(index)
  ));
  resources.green_contexts.at(index) = nullptr;

  const bool has_normal_stream =
      resources.normal_stream_id >= 0 &&
      resources.normal_stream_id <
          static_cast<int>(resources.sm_count_each_stream.size());
  if (has_normal_stream) {
    resources.sm_count_each_stream.at(
        static_cast<std::size_t>(resources.normal_stream_id)
    ) += released_sm_count;
  }
  resources.sm_count_each_stream.at(index) = 0;

  std::cout
      << "[GC released] stream[" << stream_id
      << "], returned SM=" << released_sm_count;
  if (has_normal_stream) {
    std::cout
        << ", normal stream SM="
        << resources.sm_count_each_stream.at(
            static_cast<std::size_t>(resources.normal_stream_id)
        );
  }
  std::cout << '\n';
}

inline bool release_finished_green_context_streams(
    RuntimeResources& resources
) {
  bool released_any = false;

  for (const int stream_id : resources.green_context_stream_ids) {
    const cudaStream_t stream = resources.streams.at(
        static_cast<std::size_t>(stream_id)
    );
    if (stream == nullptr) {
      continue;
    }

    const cudaError_t query_result = cudaStreamQuery(stream);
    if (query_result == cudaErrorNotReady) {
      continue;
    }

    GC_CUDA_CHECK(query_result);
    release_green_context_stream(resources, stream_id);
    released_any = true;
  }

  return released_any;
}

inline bool all_green_context_streams_released(
    const RuntimeResources& resources
) {
  for (const int stream_id : resources.green_context_stream_ids) {
    const std::size_t index = static_cast<std::size_t>(stream_id);
    if (resources.streams.at(index) != nullptr ||
        resources.green_contexts.at(index) != nullptr ||
        resources.sm_count_each_stream.at(index) != 0) {
      return false;
    }
  }

  return true;
}

inline void wait_and_release_green_context_streams(
    RuntimeResources& resources
) {
  while (!all_green_context_streams_released(resources)) {
    if (!release_finished_green_context_streams(resources)) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }
}

inline void destroy_runtime_resources(RuntimeResources& resources) {
  GC_CUDA_CHECK(cudaDeviceSynchronize());

  for (auto& stream : resources.streams) {
    if (stream != nullptr) {
      GC_CUDA_CHECK(cudaStreamDestroy(stream));
      stream = nullptr;
    }
  }
  for (auto& context : resources.green_contexts) {
    if (context != nullptr) {
      GC_CUDA_CHECK(cudaExecutionCtxDestroy(context));
      context = nullptr;
    }
  }

  resources.streams.clear();
  resources.green_contexts.clear();
  resources.resource_descriptors.clear();
  resources.sm_count_each_stream.clear();
  resources.priority_each_stream.clear();
  resources.green_context_stream_ids.clear();
  resources.normal_stream_id = -1;
}

namespace stage5_detail {

static __global__ void normal_light_kernel(
    float* data,
    int element_count,
    int iterations
) {
  const int elements_per_block =
      (element_count + gridDim.x - 1) / gridDim.x;
  const int begin = blockIdx.x * elements_per_block;
  const int end = min(begin + elements_per_block, element_count);

  if (begin >= element_count) {
    return;
  }

  for (int index = begin + threadIdx.x;
       index < end;
       index += blockDim.x) {
    float value = data[index];

    #pragma unroll 1
    for (int iteration = 0;
         iteration < iterations;
         ++iteration) {
      value = value * 1.000001f + 0.00001f;
    }

    data[index] = value;
  }
}

static __global__ void normal_heavy_kernel(
    float* data,
    int element_count,
    int iterations
) {
  const int elements_per_block =
      (element_count + gridDim.x - 1) / gridDim.x;
  const int begin = blockIdx.x * elements_per_block;
  const int end = min(begin + elements_per_block, element_count);

  if (begin >= element_count) {
    return;
  }

  for (int index = begin + threadIdx.x;
       index < end;
       index += blockDim.x) {
    float value = data[index];

    #pragma unroll 1
    for (int iteration = 0;
         iteration < iterations;
         ++iteration) {
      value = value * 1.000001f + 0.00001f;
      value = value * 0.999999f + 0.00002f;
      value = value * 1.0000003f - 0.00001f;
    }

    data[index] = value;
  }
}

inline void launch_normal_task_kernel(
    const TaskSpec& task,
    float* device_memory,
    int element_count,
    cudaStream_t stream,
    int grid_sm_count
) {
  if (task.work_units <= 0) {
    return;
  }

  constexpr int threads_per_block = 256;

  if (task.kind == KernelKind::HEAVY) {
    normal_heavy_kernel<<<
        grid_sm_count,
        threads_per_block,
        0,
        stream
    >>>(
        device_memory,
        element_count,
        task.work_units
    );
  }
  else {
    normal_light_kernel<<<
        grid_sm_count,
        threads_per_block,
        0,
        stream
    >>>(
        device_memory,
        element_count,
        task.work_units
    );
  }

  CUDA_CHECK(cudaGetLastError());
}

inline void launch_task_kernel_chunked(
    const TaskSpec& task,
    float* device_memory,
    int element_count,
    cudaStream_t stream,
    int chunk_count
) {
  if (chunk_count <= 1 || task.proc_time <= 0) {
    stg::launch_task_kernel(
        task,
        device_memory,
        element_count,
        stream
    );
    return;
  }

  const int bounded_chunk_count = std::min(
      chunk_count,
      std::max(1, task.work_units)
  );
  const int base_iterations = task.work_units / bounded_chunk_count;
  const int remainder = task.work_units % bounded_chunk_count;

  for (int chunk = 0;
       chunk < bounded_chunk_count;
       ++chunk) {
    TaskSpec chunk_task = task;
    chunk_task.work_units =
        base_iterations + (chunk < remainder ? 1 : 0);

    stg::launch_task_kernel(
        chunk_task,
        device_memory,
        element_count,
        stream
    );
  }
}

inline std::vector<std::vector<int>> make_stream_task_order(
    const std::vector<TaskSpec>& tasks,
    const StreamScheduleResult& schedule,
    int stream_count
) {
  std::vector<std::vector<int>> stream_task_order(
      static_cast<std::size_t>(stream_count)
  );

  for (const auto& task : tasks) {
    const auto stream_it = schedule.task_stream.find(task.id);
    if (stream_it == schedule.task_stream.end()) {
      throw std::runtime_error(
          "stream assignment not found for task " +
          std::to_string(task.id)
      );
    }

    const int stream_id = stream_it->second;
    if (stream_id < 0 || stream_id >= stream_count) {
      throw std::runtime_error(
          "invalid stream assignment for task " +
          std::to_string(task.id)
      );
    }

    stream_task_order.at(
        static_cast<std::size_t>(stream_id)
    ).push_back(task.id);
  }

  for (auto& task_ids : stream_task_order) {
    std::stable_sort(
        task_ids.begin(),
        task_ids.end(),
        [&](int left_id, int right_id) {
          const double left_start =
              schedule.task_start_time.at(left_id);
          const double right_start =
              schedule.task_start_time.at(right_id);

          if (left_start != right_start) {
            return left_start < right_start;
          }

          const double left_finish =
              schedule.task_finish_time.at(left_id);
          const double right_finish =
              schedule.task_finish_time.at(right_id);

          if (left_finish != right_finish) {
            return left_finish < right_finish;
          }

          return left_id < right_id;
        }
    );
  }

  return stream_task_order;
}

inline void add_taskflow_dependencies(
    const std::vector<TaskSpec>& tasks,
    const std::vector<std::vector<int>>& stream_task_order,
    std::unordered_map<int, tf::Task>& task_nodes
) {
  std::unordered_map<int, const TaskSpec*> task_by_id;
  task_by_id.reserve(tasks.size());

  for (const auto& task : tasks) {
    task_by_id.emplace(task.id, &task);

    for (const int predecessor_id : task.preds) {
      const auto predecessor_it = task_nodes.find(predecessor_id);
      const auto current_it = task_nodes.find(task.id);

      if (predecessor_it == task_nodes.end() ||
          current_it == task_nodes.end()) {
        throw std::runtime_error(
            "Taskflow dependency references an unknown task"
        );
      }

      predecessor_it->second.precede(current_it->second);
    }
  }

  for (const auto& task_ids : stream_task_order) {
    for (std::size_t index = 1;
         index < task_ids.size();
         ++index) {
      const int previous_id = task_ids.at(index - 1);
      const int current_id = task_ids.at(index);
      const TaskSpec& current_task = *task_by_id.at(current_id);
      const bool is_direct_predecessor =
          std::find(
              current_task.preds.begin(),
              current_task.preds.end(),
              previous_id
          ) != current_task.preds.end();

      if (!is_direct_predecessor) {
        task_nodes.at(previous_id).precede(
            task_nodes.at(current_id)
        );
      }
    }
  }
}

}  // namespace stage5_detail

inline GreenContextExecutionResult execute_with_green_context(
    const std::vector<TaskSpec>& tasks,
    const StreamScheduleResult& schedule,
    const std::vector<int>& stream_sm_counts,
    const GreenContextExecutionOptions& options = {}
) {
  if (tasks.empty()) {
    throw std::invalid_argument("tasks must not be empty");
  }
  if (stream_sm_counts.empty()) {
    throw std::invalid_argument("stream_sm_counts must not be empty");
  }
  validate_stream_sm_counts(stream_sm_counts);
  if (options.background_chunk_count <= 0) {
    throw std::invalid_argument(
        "background_chunk_count must be positive"
    );
  }
  if (options.scheduling_reference_sm <= 0) {
    throw std::invalid_argument(
        "scheduling_reference_sm must be positive"
    );
  }

  const int stream_count =
      static_cast<int>(stream_sm_counts.size());
  constexpr int element_count = kTaskElementCount;
  const std::size_t allocation_element_count =
      static_cast<std::size_t>(element_count) *
      static_cast<std::size_t>(stream_count);

  float* device_memory = nullptr;
  CUDA_CHECK(cudaMalloc(
      &device_memory,
      allocation_element_count * sizeof(float)
  ));
  CUDA_CHECK(cudaMemset(
      device_memory,
      0,
      allocation_element_count * sizeof(float)
  ));

  GreenContextExecutionResult result;
  RuntimeResources resources;

  {
    const auto setup_start = std::chrono::steady_clock::now();

    if (options.disable_green_context) {
      resources = create_normal_stream_resources(
          stream_count,
          options.scheduling_reference_sm
      );
    }
    else {
      resources = create_green_context_resources(stream_sm_counts);
    }

    const auto setup_end = std::chrono::steady_clock::now();
    result.resource_setup_ms =
        std::chrono::duration<double, std::milli>(
            setup_end - setup_start
        ).count();
  }

  tf::Executor executor(8);
  tf::Taskflow taskflow;
  std::unordered_map<int, tf::Task> task_nodes;
  std::unordered_map<int, cudaEvent_t> done_events;
  std::unordered_map<int, cudaEvent_t> kernel_start_events;
  std::unordered_map<int, cudaEvent_t> kernel_stop_events;

  task_nodes.reserve(tasks.size());
  done_events.reserve(tasks.size());
  kernel_start_events.reserve(tasks.size());
  kernel_stop_events.reserve(tasks.size());

  cudaStream_t timing_stream = nullptr;
  cudaEvent_t origin_event = nullptr;

  CUDA_CHECK(cudaStreamCreateWithFlags(
      &timing_stream,
      cudaStreamNonBlocking
  ));
  CUDA_CHECK(cudaEventCreate(&origin_event));
  CUDA_CHECK(cudaEventRecord(origin_event, timing_stream));
  CUDA_CHECK(cudaEventSynchronize(origin_event));

  for (const auto& task : tasks) {
    cudaEvent_t done_event = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    CUDA_CHECK(cudaEventCreateWithFlags(
        &done_event,
        cudaEventDisableTiming
    ));
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    done_events.emplace(task.id, done_event);
    kernel_start_events.emplace(task.id, start_event);
    kernel_stop_events.emplace(task.id, stop_event);
  }

  for (const auto& task : tasks) {
    tf::Task node = taskflow.emplace([&, task]() {
      const int stream_id = schedule.task_stream.at(task.id);
      cudaStream_t stream = get_stream_by_id(resources, stream_id);

      for (const int predecessor_id : task.preds) {
        const auto event_it = done_events.find(predecessor_id);
        if (event_it == done_events.end()) {
          throw std::runtime_error(
              "completion event not found for predecessor " +
              std::to_string(predecessor_id)
          );
        }

        CUDA_CHECK(cudaStreamWaitEvent(stream, event_it->second, 0));
      }

      CUDA_CHECK(cudaEventRecord(
          kernel_start_events.at(task.id),
          stream
      ));

      float* task_memory =
          device_memory +
          static_cast<std::size_t>(stream_id) *
              static_cast<std::size_t>(element_count);

      if (!options.disable_green_context &&
          stream_count >= 2 &&
          stream_id == 0) {
        stage5_detail::launch_normal_task_kernel(
            task,
            task_memory,
            element_count,
            stream,
            options.scheduling_reference_sm
        );
      }
      else {
        const int chunk_count =
            options.disable_green_context && stream_id != 0
                ? options.background_chunk_count
                : 1;

        stage5_detail::launch_task_kernel_chunked(
            task,
            task_memory,
            element_count,
            stream,
            chunk_count
        );
      }

      CUDA_CHECK(cudaEventRecord(
          kernel_stop_events.at(task.id),
          stream
      ));
      CUDA_CHECK(cudaEventRecord(done_events.at(task.id), stream));
    });

    node.name(std::to_string(task.id));
    task_nodes.emplace(task.id, node);
  }

  const auto stream_task_order =
      stage5_detail::make_stream_task_order(
          tasks,
          schedule,
          stream_count
      );
  stage5_detail::add_taskflow_dependencies(
      tasks,
      stream_task_order,
      task_nodes
  );

  const auto submit_start = std::chrono::steady_clock::now();
  executor.run(taskflow).wait();

  if (!options.disable_green_context && stream_count >= 2) {
    wait_and_release_green_context_streams(resources);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  const auto submit_end = std::chrono::steady_clock::now();

  result.gpu_submit_wait_ms =
      std::chrono::duration<double, std::milli>(
          submit_end - submit_start
      ).count();

  double first_kernel_start_ms =
      std::numeric_limits<double>::max();
  double last_kernel_stop_ms = 0.0;

  for (const auto& task : tasks) {
    float start_from_origin_ms = 0.0f;
    float stop_from_origin_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &start_from_origin_ms,
        origin_event,
        kernel_start_events.at(task.id)
    ));
    CUDA_CHECK(cudaEventElapsedTime(
        &stop_from_origin_ms,
        origin_event,
        kernel_stop_events.at(task.id)
    ));

    first_kernel_start_ms = std::min(
        first_kernel_start_ms,
        static_cast<double>(start_from_origin_ms)
    );
    last_kernel_stop_ms = std::max(
        last_kernel_stop_ms,
        static_cast<double>(stop_from_origin_ms)
    );
  }

  result.gpu_kernel_ms =
      last_kernel_stop_ms - first_kernel_start_ms;

  print_runtime_resource_state(
      resources,
      "after completed GC release"
  );

  if (all_green_context_streams_released(resources)) {
    std::cout << "[CHECK] all GC streams released\n";
  }
  else {
    std::cout << "[CHECK] some GC streams are still alive\n";
  }

  for (auto& [task_id, event] : done_events) {
    (void)task_id;
    CUDA_CHECK(cudaEventDestroy(event));
  }
  for (auto& [task_id, event] : kernel_start_events) {
    (void)task_id;
    CUDA_CHECK(cudaEventDestroy(event));
  }
  for (auto& [task_id, event] : kernel_stop_events) {
    (void)task_id;
    CUDA_CHECK(cudaEventDestroy(event));
  }

  CUDA_CHECK(cudaEventDestroy(origin_event));
  CUDA_CHECK(cudaStreamDestroy(timing_stream));
  destroy_runtime_resources(resources);
  CUDA_CHECK(cudaFree(device_memory));

  return result;
}

#undef GC_CUDA_CHECK
