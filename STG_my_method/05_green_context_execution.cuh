#pragma once

#include <cuda_runtime.h>
#include <taskflow/taskflow.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "00_pipeline_configuration.hpp"
#include "01_stg_analysis.hpp"
#include "03_stream_assignment.hpp"
#include "04_execution_configuration_selection.hpp"

/*
 * ================================================================
 * Stage 5
 *
 * Stage 4で決定した
 *
 *   ・SM配分
 *   ・Stream配置
 *
 * に従ってGPU上で実行する。
 *
 * GC Streamの全タスクが完了した場合は、
 * そのGCを実行途中で解放する。
 *
 * 解放されたSMはStream 0へ戻し、
 * その後に起動するStream 0のタスクで利用する。
 *
 *
 * 例:
 *
 *   初期
 *     Stream 0 : 82 SM
 *     Stream 1 : 24 SM
 *     Stream 2 :  8 SM
 *
 *                ↓
 *
 *   Stream 2終了
 *     Stream 0 : 90 SM
 *
 *                ↓
 *
 *   Stream 1終了
 *     Stream 0 : 114 SM
 *
 *
 * また、SM数に応じて
 *
 *   grid block数
 *   threads/block
 *
 * を調整する。
 *
 * kTaskElementCount = 64 * 256 の仕事量は変更しない。
 * ================================================================
 */


struct GreenContextExecutionOptions {
  bool disable_green_context = false;

  int background_chunk_count = 1;

  int scheduling_reference_sm =
      kSchedulingReferenceSmCount;
};


struct GreenContextExecutionResult {
  double resource_setup_ms = 0.0;

  double gpu_submit_wait_ms = 0.0;

  double gpu_kernel_ms = 0.0;
};


#define GC_CUDA_CHECK(expression)                                           \
  do {                                                                      \
    const cudaError_t gc_error__ = (expression);                            \
    if (gc_error__ != cudaSuccess) {                                        \
      throw std::runtime_error(                                             \
          std::string("CUDA error: ") +                                     \
          cudaGetErrorString(gc_error__) +                                  \
          " at " + __FILE__ + ":" +                                         \
          std::to_string(__LINE__)                                          \
      );                                                                    \
    }                                                                       \
  } while (0)


/*
 * ================================================================
 * GPU SM分割情報
 * ================================================================
 */
struct SmPartitionInfo {
  int available_sm = 0;

  int unit_sm = 0;

  int min_group_sm = 0;
};


/*
 * ================================================================
 * 実行時CUDA資源
 * ================================================================
 */
struct RuntimeResources {
  std::vector<cudaStream_t>
      streams;

  std::vector<cudaExecutionContext_t>
      green_contexts;

  std::vector<cudaDevResourceDesc_t>
      resource_descriptors;

  /*
   * 現在各Streamが持っているSM数。
   *
   * GC解放後:
   *
   *   GC側      -> 0
   *   Stream 0  -> 解放SMを加算
   */
  std::vector<int>
      sm_count_each_stream;

  std::vector<long long>
      priority_each_stream;

  int normal_stream_id = 0;

  std::vector<int>
      green_context_stream_ids;
};


/*
 * ================================================================
 * alignment
 * ================================================================
 */
inline int align_up_int(
    int value,
    int alignment
) {
  if (alignment <= 0) {
    throw std::invalid_argument(
        "alignment must be positive"
    );
  }

  return
      ((value + alignment - 1) /
       alignment) *
      alignment;
}


/*
 * ================================================================
 * GPUのSM分割情報取得
 * ================================================================
 */
inline SmPartitionInfo query_sm_partition_info(
    int gpu_device_index = 0
) {
  GC_CUDA_CHECK(
      cudaSetDevice(
          gpu_device_index
      )
  );


  cudaDevResource initial_sm_resource{};


  GC_CUDA_CHECK(
      cudaDeviceGetDevResource(
          gpu_device_index,
          &initial_sm_resource,
          cudaDevResourceTypeSm
      )
  );


  const int available_sm =
      static_cast<int>(
          initial_sm_resource.sm.smCount
      );


  const int min_partition_size =
      static_cast<int>(
          initial_sm_resource.sm.minSmPartitionSize
      );


  const int coscheduled_alignment =
      static_cast<int>(
          initial_sm_resource.sm.smCoscheduledAlignment
      );


  const int unit_sm =
      std::max(
          2,
          coscheduled_alignment
      );


  const int min_group_sm =
      align_up_int(
          std::max(
              2,
              min_partition_size
          ),
          unit_sm
      );


  if (
      available_sm <= 0 ||
      unit_sm <= 0 ||
      min_group_sm <= 0
  ) {
    throw std::runtime_error(
        "invalid SM partition information"
    );
  }


  return {
      available_sm,
      unit_sm,
      min_group_sm
  };
}


/*
 * ================================================================
 * 114 SM GPUか確認
 * ================================================================
 */
inline void validate_fixed_sm_table_compatibility(
    const SmPartitionInfo& partition_info
) {
  if (
      partition_info.available_sm !=
      kSchedulingReferenceSmCount
  ) {
    throw std::runtime_error(
        "fixed SM table requires a 114-SM GPU"
    );
  }
}


/*
 * ================================================================
 * GCとして実現可能なSM配分か確認
 * ================================================================
 */
inline void validate_green_context_sm_counts(
    const std::vector<int>& stream_sm_counts,
    const SmPartitionInfo& partition_info
) {
  validate_stream_sm_counts(
      stream_sm_counts
  );


  /*
   * 1 Stream
   */
  if (
      stream_sm_counts.size() == 1
  ) {
    if (
        stream_sm_counts.front() !=
        partition_info.available_sm
    ) {
      throw std::invalid_argument(
          "single normal stream must use all available SMs"
      );
    }


    return;
  }


  int gc_total =
      0;


  /*
   * Stream 1以降はGC
   */
  for (
      std::size_t stream_id = 1;
      stream_id < stream_sm_counts.size();
      ++stream_id
  ) {
    const int sm_count =
        stream_sm_counts.at(
            stream_id
        );


    if (
        sm_count <
        partition_info.min_group_sm
    ) {
      throw std::invalid_argument(
          "GC SM count is smaller than minimum: stream " +
          std::to_string(stream_id)
      );
    }


    if (
        sm_count %
            partition_info.unit_sm !=
        0
    ) {
      throw std::invalid_argument(
          "GC SM count is not aligned: stream " +
          std::to_string(stream_id)
      );
    }


    gc_total +=
        sm_count;
  }


  /*
   * GC分割後にprimary contextへ残るSM数
   */
  const int normal_sm =
      partition_info.available_sm -
      gc_total;


  if (
      normal_sm <= 0
  ) {
    throw std::invalid_argument(
        "Green Contexts must leave SMs for the normal stream"
    );
  }


  /*
   * Stage 2～4で予測したStream 0のSM数と
   * 実際に残るSM数を一致させる。
   */
  if (
      stream_sm_counts.front() !=
      normal_sm
  ) {
    throw std::invalid_argument(
        "stream 0 SM count must equal "
        "the SMs left after GC partitioning"
    );
  }
}


/*
 * ================================================================
 * Stream取得
 * ================================================================
 */
inline cudaStream_t get_stream_by_id(
    const RuntimeResources& resources,
    int stream_id
) {
  if (
      stream_id < 0 ||
      stream_id >=
          static_cast<int>(
              resources.streams.size()
          )
  ) {
    throw std::runtime_error(
        "invalid stream id: " +
        std::to_string(
            stream_id
        )
    );
  }


  const cudaStream_t stream =
      resources.streams.at(
          static_cast<std::size_t>(
              stream_id
          )
      );


  if (
      stream ==
      nullptr
  ) {
    throw std::runtime_error(
        "stream is not available: " +
        std::to_string(
            stream_id
        )
    );
  }


  return stream;
}


/*
 * ================================================================
 * GC Stream判定
 * ================================================================
 */
inline bool is_green_context_stream_id(
    const RuntimeResources& resources,
    int stream_id
) {
  return
      std::find(
          resources.green_context_stream_ids.begin(),
          resources.green_context_stream_ids.end(),
          stream_id
      ) !=
      resources.green_context_stream_ids.end();
}


/*
 * ================================================================
 * Stage 3配置結果の検証
 * ================================================================
 */
inline void validate_execution_schedule(
    const std::vector<TaskSpec>& tasks,

    const StreamScheduleResult& schedule,

    int stream_count
) {
  if (
      stream_count <= 0
  ) {
    throw std::invalid_argument(
        "stream_count must be positive"
    );
  }


  for (
      const auto& task :
      tasks
  ) {
    const auto stream_it =
        schedule.task_stream.find(
            task.id
        );


    if (
        stream_it ==
        schedule.task_stream.end()
    ) {
      throw std::invalid_argument(
          "stream assignment not found for task: " +
          std::to_string(
              task.id
          )
      );
    }


    if (
        stream_it->second < 0 ||
        stream_it->second >=
            stream_count
    ) {
      throw std::invalid_argument(
          "invalid stream assignment for task: " +
          std::to_string(
              task.id
          )
      );
    }


    if (
        schedule.task_start_time.find(
            task.id
        ) ==
            schedule.task_start_time.end() ||

        schedule.task_finish_time.find(
            task.id
        ) ==
            schedule.task_finish_time.end()
    ) {
      throw std::invalid_argument(
          "predicted schedule time not found for task: " +
          std::to_string(
              task.id
          )
      );
    }
  }
}


/*
 * ================================================================
 * GCなし比較用Stream
 * ================================================================
 */
inline RuntimeResources create_normal_stream_resources(
    int stream_count,

    int scheduling_reference_sm
) {
  if (
      stream_count <= 0 ||
      scheduling_reference_sm <= 0
  ) {
    throw std::invalid_argument(
        "stream_count and scheduling_reference_sm "
        "must be positive"
    );
  }


  GC_CUDA_CHECK(
      cudaSetDevice(0)
  );


  RuntimeResources resources;


  resources.streams.resize(
      static_cast<std::size_t>(
          stream_count
      ),
      nullptr
  );


  resources.green_contexts.resize(
      static_cast<std::size_t>(
          stream_count
      ),
      nullptr
  );


  resources.resource_descriptors.resize(
      static_cast<std::size_t>(
          stream_count
      ),
      nullptr
  );


  resources.sm_count_each_stream.resize(
      static_cast<std::size_t>(
          stream_count
      ),
      scheduling_reference_sm
  );


  resources.priority_each_stream.resize(
      static_cast<std::size_t>(
          stream_count
      ),
      0
  );


  resources.normal_stream_id =
      0;


  int least_priority =
      0;


  int greatest_priority =
      0;


  GC_CUDA_CHECK(
      cudaDeviceGetStreamPriorityRange(
          &least_priority,
          &greatest_priority
      )
  );


  for (
      int stream_id = 0;
      stream_id < stream_count;
      ++stream_id
  ) {
    const int priority =
        stream_id == 0
            ? greatest_priority
            : least_priority;


    GC_CUDA_CHECK(
        cudaStreamCreateWithPriority(
            &resources.streams.at(
                static_cast<std::size_t>(
                    stream_id
                )
            ),
            cudaStreamNonBlocking,
            priority
        )
    );


    resources.priority_each_stream.at(
        static_cast<std::size_t>(
            stream_id
        )
    ) = priority;
  }


  std::cout
      << "[normal stream mode] no Green Context\n";


  return resources;
}


/*
 * ================================================================
 * Stage 4で決定したSM配分からGCを構築
 * ================================================================
 */
inline RuntimeResources create_green_context_resources(
    const std::vector<int>& stream_sm_counts,

    int gpu_device_index = 0
) {
  if (
      stream_sm_counts.empty()
  ) {
    throw std::invalid_argument(
        "at least one stream is required"
    );
  }


  GC_CUDA_CHECK(
      cudaSetDevice(
          gpu_device_index
      )
  );


  const SmPartitionInfo partition_info =
      query_sm_partition_info(
          gpu_device_index
      );


  validate_fixed_sm_table_compatibility(
      partition_info
  );


  validate_green_context_sm_counts(
      stream_sm_counts,
      partition_info
  );


  const int stream_count =
      static_cast<int>(
          stream_sm_counts.size()
      );


  const int gc_count =
      stream_count - 1;


  RuntimeResources resources;


  resources.streams.resize(
      static_cast<std::size_t>(
          stream_count
      ),
      nullptr
  );


  resources.green_contexts.resize(
      static_cast<std::size_t>(
          stream_count
      ),
      nullptr
  );


  resources.resource_descriptors.resize(
      static_cast<std::size_t>(
          stream_count
      ),
      nullptr
  );


  resources.sm_count_each_stream =
      stream_sm_counts;


  resources.priority_each_stream.resize(
      static_cast<std::size_t>(
          stream_count
      ),
      0
  );


  resources.normal_stream_id =
      0;


  for (
      int stream_id = 1;
      stream_id < stream_count;
      ++stream_id
  ) {
    resources.green_context_stream_ids.push_back(
        stream_id
    );
  }


  /*
   * 1 Stream
   */
  if (
      stream_count == 1
  ) {
    GC_CUDA_CHECK(
        cudaStreamCreateWithFlags(
            &resources.streams.at(0),
            cudaStreamNonBlocking
        )
    );


    return resources;
  }


  cudaDevResource initial_sm_resource{};


  GC_CUDA_CHECK(
      cudaDeviceGetDevResource(
          gpu_device_index,
          &initial_sm_resource,
          cudaDevResourceTypeSm
      )
  );


  std::vector<
      cudaDevResource
  > split_sm_resources(
      static_cast<std::size_t>(
          gc_count
      )
  );


  std::vector<
      cudaDevSmResourceGroupParams
  > group_parameters(
      static_cast<std::size_t>(
          gc_count
      )
  );


  /*
   * GCごとのSM数
   */
  for (
      int gc_index = 0;
      gc_index < gc_count;
      ++gc_index
  ) {
    const int stream_id =
        gc_index + 1;


    auto& parameters =
        group_parameters.at(
            static_cast<std::size_t>(
                gc_index
            )
        );


    parameters =
        {};


    parameters.smCount =
        static_cast<unsigned int>(
            stream_sm_counts.at(
                static_cast<std::size_t>(
                    stream_id
                )
            )
        );


    parameters.coscheduledSmCount =
        2;


    parameters.preferredCoscheduledSmCount =
        2;


    parameters.flags =
        0;
  }


  GC_CUDA_CHECK(
      cudaDeviceSynchronize()
  );


  /*
   * SM resource分割
   */
  const cudaError_t split_error =
      cudaDevSmResourceSplit(
          split_sm_resources.data(),

          static_cast<unsigned int>(
              gc_count
          ),

          &initial_sm_resource,

          nullptr,

          0,

          group_parameters.data()
      );


  if (
      split_error !=
      cudaSuccess
  ) {
    std::cerr
        << "cudaDevSmResourceSplit failed\n";


    GC_CUDA_CHECK(
        split_error
    );
  }


  /*
   * Descriptor
   */
  for (
      int gc_index = 0;
      gc_index < gc_count;
      ++gc_index
  ) {
    const int stream_id =
        gc_index + 1;


    GC_CUDA_CHECK(
        cudaDevResourceGenerateDesc(
            &resources.resource_descriptors.at(
                static_cast<std::size_t>(
                    stream_id
                )
            ),

            &split_sm_resources.at(
                static_cast<std::size_t>(
                    gc_index
                )
            ),

            1
        )
    );
  }


  /*
   * GC + Stream
   */
  for (
      int stream_id = 1;
      stream_id < stream_count;
      ++stream_id
  ) {
    GC_CUDA_CHECK(
        cudaGreenCtxCreate(
            &resources.green_contexts.at(
                static_cast<std::size_t>(
                    stream_id
                )
            ),

            resources.resource_descriptors.at(
                static_cast<std::size_t>(
                    stream_id
                )
            ),

            gpu_device_index,

            0
        )
    );


    GC_CUDA_CHECK(
        cudaExecutionCtxStreamCreate(
            &resources.streams.at(
                static_cast<std::size_t>(
                    stream_id
                )
            ),

            resources.green_contexts.at(
                static_cast<std::size_t>(
                    stream_id
                )
            ),

            cudaStreamNonBlocking,

            0
        )
    );
  }


  /*
   * Stream 0
   */
  GC_CUDA_CHECK(
      cudaStreamCreateWithFlags(
          &resources.streams.at(0),
          cudaStreamNonBlocking
      )
  );


  std::cout
      << "[GC setup] selected SM allocation: ";


  for (
      std::size_t index = 0;
      index < stream_sm_counts.size();
      ++index
  ) {
    if (
        index != 0
    ) {
      std::cout
          << " / ";
    }


    std::cout
        << stream_sm_counts.at(
            index
        );
  }


  std::cout
      << '\n';


  return resources;
}


/*
 * ================================================================
 * Runtime状態表示
 * ================================================================
 */
inline void print_runtime_resource_state(
    const RuntimeResources& resources,

    const std::string& tag
) {
  std::cout
      << "\n===== Runtime Resource State: "
      << tag
      << " =====\n";


  for (
      int stream_id = 0;
      stream_id <
          static_cast<int>(
              resources.streams.size()
          );
      ++stream_id
  ) {
    const std::size_t index =
        static_cast<std::size_t>(
            stream_id
        );


    std::cout
        << "stream["
        << stream_id
        << "] ";


    if (
        stream_id ==
        resources.normal_stream_id
    ) {
      std::cout
          << "[normal] ";
    }

    else if (
        is_green_context_stream_id(
            resources,
            stream_id
        )
    ) {
      std::cout
          << "[GC] ";
    }

    else {
      std::cout
          << "[shared] ";
    }


    std::cout
        << "SM="
        << resources.sm_count_each_stream.at(
            index
        )

        << ", stream="
        << resources.streams.at(
            index
        )

        << ", green_ctx="
        << resources.green_contexts.at(
            index
        )

        << '\n';
  }


  std::cout
      << "=============================================\n";
}


/*
 * ================================================================
 * 現在Stream 0が利用可能なSM数
 * ================================================================
 */
inline int get_current_normal_sm_count(
    const RuntimeResources& resources,

    std::mutex& resource_mutex
) {
  std::lock_guard<std::mutex> lock(
      resource_mutex
  );


  if (
      resources.normal_stream_id < 0 ||

      resources.normal_stream_id >=
          static_cast<int>(
              resources.sm_count_each_stream.size()
          )
  ) {
    throw std::runtime_error(
        "normal stream is not available"
    );
  }


  return
      resources.sm_count_each_stream.at(
          static_cast<std::size_t>(
              resources.normal_stream_id
          )
      );
}


/*
 * ================================================================
 * GC解放
 * ================================================================
 */
inline void release_green_context_stream(
    RuntimeResources& resources,

    int stream_id,

    std::mutex& resource_mutex
) {
  std::lock_guard<std::mutex> lock(
      resource_mutex
  );


  if (
      !is_green_context_stream_id(
          resources,
          stream_id
      )
  ) {
    return;
  }


  const std::size_t index =
      static_cast<std::size_t>(
          stream_id
      );


  /*
   * 既に解放済み
   */
  if (
      resources.green_contexts.at(index) ==
      nullptr
  ) {
    return;
  }


  const int released_sm =
      resources.sm_count_each_stream.at(
          index
      );


  /*
   * GC Stream完了確認
   */
  if (
      resources.streams.at(index) !=
      nullptr
  ) {
    GC_CUDA_CHECK(
        cudaStreamSynchronize(
            resources.streams.at(
                index
            )
        )
    );


    GC_CUDA_CHECK(
        cudaStreamDestroy(
            resources.streams.at(
                index
            )
        )
    );


    resources.streams.at(
        index
    ) =
        nullptr;
  }


  /*
   * GC破棄
   */
  GC_CUDA_CHECK(
      cudaExecutionCtxDestroy(
          resources.green_contexts.at(
              index
          )
      )
  );


  resources.green_contexts.at(
      index
  ) =
      nullptr;


  /*
   * bookkeeping:
   * 解放SMをStream 0へ戻す。
   */
  if (
      resources.normal_stream_id >= 0
  ) {
    resources.sm_count_each_stream.at(
        static_cast<std::size_t>(
            resources.normal_stream_id
        )
    ) +=
        released_sm;
  }


  resources.sm_count_each_stream.at(
      index
  ) =
      0;


  std::cout
      << "[GC released] stream["
      << stream_id
      << "]"
      << ", returned SM="
      << released_sm;


  if (
      resources.normal_stream_id >= 0
  ) {
    std::cout
        << ", current normal SM="
        << resources.sm_count_each_stream.at(
            static_cast<std::size_t>(
                resources.normal_stream_id
            )
        );
  }


  std::cout
      << '\n';
}


/*
 * ================================================================
 * cleanup用GC解放確認
 * ================================================================
 */
inline bool release_finished_green_context_streams(
    RuntimeResources& resources,

    std::mutex& resource_mutex
) {
  bool released_any =
      false;


  for (
      const int stream_id :
      resources.green_context_stream_ids
  ) {
    cudaStream_t stream =
        nullptr;


    {
      std::lock_guard<std::mutex> lock(
          resource_mutex
      );


      stream =
          resources.streams.at(
              static_cast<std::size_t>(
                  stream_id
              )
          );
    }


    if (
        stream ==
        nullptr
    ) {
      continue;
    }


    const cudaError_t query_result =
        cudaStreamQuery(
            stream
        );


    if (
        query_result ==
        cudaErrorNotReady
    ) {
      continue;
    }


    GC_CUDA_CHECK(
        query_result
    );


    release_green_context_stream(
        resources,
        stream_id,
        resource_mutex
    );


    released_any =
        true;
  }


  return released_any;
}


inline bool all_green_context_streams_released(
    const RuntimeResources& resources,

    std::mutex& resource_mutex
) {
  std::lock_guard<std::mutex> lock(
      resource_mutex
  );


  for (
      const int stream_id :
      resources.green_context_stream_ids
  ) {
    const std::size_t index =
        static_cast<std::size_t>(
            stream_id
        );


    if (
        resources.streams.at(index) !=
            nullptr ||

        resources.green_contexts.at(index) !=
            nullptr ||

        resources.sm_count_each_stream.at(index) !=
            0
    ) {
      return false;
    }
  }


  return true;
}


inline void wait_and_release_green_context_streams(
    RuntimeResources& resources,

    std::mutex& resource_mutex
) {
  while (
      !all_green_context_streams_released(
          resources,
          resource_mutex
      )
  ) {
    if (
        !release_finished_green_context_streams(
            resources,
            resource_mutex
        )
    ) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(
              50
          )
      );
    }
  }
}


/*
 * ================================================================
 * Runtime Resource破棄
 * ================================================================
 */
inline void destroy_runtime_resources(
    RuntimeResources& resources
) {
  GC_CUDA_CHECK(
      cudaDeviceSynchronize()
  );


  for (
      auto& stream :
      resources.streams
  ) {
    if (
        stream !=
        nullptr
    ) {
      GC_CUDA_CHECK(
          cudaStreamDestroy(
              stream
          )
      );


      stream =
          nullptr;
    }
  }


  for (
      auto& context :
      resources.green_contexts
  ) {
    if (
        context !=
        nullptr
    ) {
      GC_CUDA_CHECK(
          cudaExecutionCtxDestroy(
              context
          )
      );


      context =
          nullptr;
    }
  }


  resources.streams.clear();

  resources.green_contexts.clear();

  resources.resource_descriptors.clear();

  resources.sm_count_each_stream.clear();

  resources.priority_each_stream.clear();

  resources.green_context_stream_ids.clear();

  resources.normal_stream_id =
      -1;
}


namespace stage5_detail {


/*
 * ================================================================
 * LIGHT kernel
 * ================================================================
 */
static __global__ void configured_light_kernel(
    float* data,

    int element_count,

    int iterations
) {
  const int elements_per_block =
      (
          element_count +
          gridDim.x -
          1
      ) /
      gridDim.x;


  const int begin =
      blockIdx.x *
      elements_per_block;


  const int end =
      min(
          begin +
          elements_per_block,
          element_count
      );


  if (
      begin >=
      element_count
  ) {
    return;
  }


  for (
      int index =
          begin +
          threadIdx.x;

      index <
          end;

      index +=
          blockDim.x
  ) {
    float value =
        data[index];


    #pragma unroll 1
    for (
        int iteration = 0;
        iteration < iterations;
        ++iteration
    ) {
      value =
          value *
              1.000001f +
          0.00001f;
    }


    data[index] =
        value;
  }
}


/*
 * ================================================================
 * HEAVY kernel
 * ================================================================
 */
static __global__ void configured_heavy_kernel(
    float* data,

    int element_count,

    int iterations
) {
  const int elements_per_block =
      (
          element_count +
          gridDim.x -
          1
      ) /
      gridDim.x;


  const int begin =
      blockIdx.x *
      elements_per_block;


  const int end =
      min(
          begin +
          elements_per_block,
          element_count
      );


  if (
      begin >=
      element_count
  ) {
    return;
  }


  for (
      int index =
          begin +
          threadIdx.x;

      index <
          end;

      index +=
          blockDim.x
  ) {
    float value =
        data[index];


    #pragma unroll 1
    for (
        int iteration = 0;
        iteration < iterations;
        ++iteration
    ) {
      value =
          value *
              1.000001f +
          0.00001f;


      value =
          value *
              0.999999f +
          0.00002f;


      value =
          value *
              1.0000003f -
          0.00001f;
    }


    data[index] =
        value;
  }
}


/*
 * ================================================================
 * SM数から起動block数を求める
 * ================================================================
 *
 * 原則:
 *
 *   1 SMあたり1 block
 *
 * とする。
 *
 * task.parallel_sm_limitも上限として使用する。
 */
inline int calculate_grid_block_count(
    const TaskSpec& task,

    int configured_stream_sm
) {
  if (
      configured_stream_sm <= 0
  ) {
    throw std::invalid_argument(
        "configured_stream_sm must be positive"
    );
  }


  return
      std::max(
          1,

          std::min(
              configured_stream_sm,

              std::max(
                  1,
                  task.parallel_sm_limit
              )
          )
      );
}


/*
 * ================================================================
 * grid数に合わせてthreads/blockを計算
 * ================================================================
 *
 * 従来:
 *
 *   常に256 threads
 *
 *
 * 今回:
 *
 *   各blockが担当する要素数を求め、
 *   warp単位に切り上げる。
 *
 *
 * element_count = 16384 の場合:
 *
 * 64 blocks
 *   256 elements/block
 *   -> 256 threads
 *
 * 82 blocks
 *   約200 elements/block
 *   -> 224 threads
 *
 * 90 blocks
 *   約183 elements/block
 *   -> 192 threads
 *
 * 114 blocks
 *   約144 elements/block
 *   -> 160 threads
 *
 *
 * これにより114 blocks時に
 * 114 * 256 threadsを起動する無駄を減らす。
 */
inline int calculate_threads_per_block(
    int element_count,

    int grid_block_count
) {
  if (
      element_count <= 0
  ) {
    throw std::invalid_argument(
        "element_count must be positive"
    );
  }


  if (
      grid_block_count <= 0
  ) {
    throw std::invalid_argument(
        "grid_block_count must be positive"
    );
  }


  constexpr int warp_size =
      32;


  constexpr int maximum_threads_per_block =
      256;


  const int elements_per_block =
      (
          element_count +
          grid_block_count -
          1
      ) /
      grid_block_count;


  /*
   * warp単位へ切り上げる。
   */
  const int warp_aligned_threads =
      (
          (
              elements_per_block +
              warp_size -
              1
          ) /
          warp_size
      ) *
      warp_size;


  return
      std::max(
          warp_size,

          std::min(
              maximum_threads_per_block,
              warp_aligned_threads
          )
      );
}


/*
 * ================================================================
 * SM数を考慮したkernel起動
 * ================================================================
 *
 * Stream 0だけでなくGC Streamでも使用する。
 *
 *
 * 例:
 *
 * configured_stream_sm = 8
 *
 *   grid = 8
 *   threads = 256
 *
 *
 * configured_stream_sm = 82
 *
 *   grid = 82
 *   threads = 224
 *
 *
 * configured_stream_sm = 114
 *
 *   grid = 114
 *   threads = 160
 *
 *
 * 仕事量kTaskElementCountは変えない。
 * ================================================================
 */
inline void launch_configured_task_kernel(
    const TaskSpec& task,

    float* device_memory,

    int element_count,

    cudaStream_t stream,

    int configured_stream_sm
) {
  if (
      task.work_units <= 0
  ) {
    return;
  }


  const int grid_block_count =
      calculate_grid_block_count(
          task,
          configured_stream_sm
      );


  const int threads_per_block =
      calculate_threads_per_block(
          element_count,
          grid_block_count
      );


  if (
      task.kind ==
      KernelKind::HEAVY
  ) {
    configured_heavy_kernel<<<
        grid_block_count,
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
    configured_light_kernel<<<
        grid_block_count,
        threads_per_block,
        0,
        stream
    >>>(
        device_memory,
        element_count,
        task.work_units
    );
  }


  GC_CUDA_CHECK(
      cudaGetLastError()
  );
}


/*
 * ================================================================
 * chunk実行
 * ================================================================
 *
 * GCなし比較などで使用する。
 *
 * SM数・grid/thread構成は
 * launch_configured_task_kernel()へ統一する。
 * ================================================================
 */
inline void launch_task_kernel_chunked(
    const TaskSpec& task,

    float* device_memory,

    int element_count,

    cudaStream_t stream,

    int chunk_count,

    int configured_stream_sm
) {
  if (
      chunk_count <= 1 ||
      task.proc_time <= 0
  ) {
    launch_configured_task_kernel(
        task,
        device_memory,
        element_count,
        stream,
        configured_stream_sm
    );


    return;
  }


  const int bounded_chunk_count =
      std::min(
          chunk_count,

          std::max(
              1,
              task.work_units
          )
      );


  const int base_iterations =
      task.work_units /
      bounded_chunk_count;


  const int remainder =
      task.work_units %
      bounded_chunk_count;


  for (
      int chunk = 0;
      chunk < bounded_chunk_count;
      ++chunk
  ) {
    TaskSpec chunk_task =
        task;


    chunk_task.work_units =
        base_iterations +
        (
            chunk <
                remainder
                ? 1
                : 0
        );


    launch_configured_task_kernel(
        chunk_task,
        device_memory,
        element_count,
        stream,
        configured_stream_sm
    );
  }
}


/*
 * ================================================================
 * Streamごとのタスク順序
 * ================================================================
 */
inline std::vector<std::vector<int>>
make_stream_task_order(
    const std::vector<TaskSpec>& tasks,

    const StreamScheduleResult& schedule,

    int stream_count
) {
  std::vector<
      std::vector<int>
  > stream_task_order(
      static_cast<std::size_t>(
          stream_count
      )
  );


  for (
      const auto& task :
      tasks
  ) {
    const auto stream_it =
        schedule.task_stream.find(
            task.id
        );


    if (
        stream_it ==
        schedule.task_stream.end()
    ) {
      throw std::runtime_error(
          "stream assignment not found for task " +
          std::to_string(
              task.id
          )
      );
    }


    const int stream_id =
        stream_it->second;


    if (
        stream_id < 0 ||
        stream_id >=
            stream_count
    ) {
      throw std::runtime_error(
          "invalid stream assignment for task " +
          std::to_string(
              task.id
          )
      );
    }


    stream_task_order.at(
        static_cast<std::size_t>(
            stream_id
        )
    ).push_back(
        task.id
    );
  }


  /*
   * Stage 3の予測開始時刻順
   */
  for (
      auto& task_ids :
      stream_task_order
  ) {
    std::stable_sort(
        task_ids.begin(),
        task_ids.end(),

        [&](
            int left_id,
            int right_id
        ) {
          const double left_start =
              schedule.task_start_time.at(
                  left_id
              );


          const double right_start =
              schedule.task_start_time.at(
                  right_id
              );


          if (
              left_start !=
              right_start
          ) {
            return
                left_start <
                right_start;
          }


          const double left_finish =
              schedule.task_finish_time.at(
                  left_id
              );


          const double right_finish =
              schedule.task_finish_time.at(
                  right_id
              );


          if (
              left_finish !=
              right_finish
          ) {
            return
                left_finish <
                right_finish;
          }


          return
              left_id <
              right_id;
        }
    );
  }


  return stream_task_order;
}


/*
 * ================================================================
 * Streamごとの最後のタスク
 * ================================================================
 */
inline std::vector<int>
make_last_task_of_stream(
    const std::vector<
        std::vector<int>
    >& stream_task_order
) {
  std::vector<int>
      last_task_of_stream(
          stream_task_order.size(),
          -1
      );


  for (
      std::size_t stream_id = 0;
      stream_id < stream_task_order.size();
      ++stream_id
  ) {
    const auto& task_ids =
        stream_task_order.at(
            stream_id
        );


    if (
        !task_ids.empty()
    ) {
      last_task_of_stream.at(
          stream_id
      ) =
          task_ids.back();
    }
  }


  return last_task_of_stream;
}


/*
 * ================================================================
 * Taskflow依存
 * ================================================================
 */
inline void add_taskflow_dependencies(
    const std::vector<TaskSpec>& tasks,

    const std::vector<
        std::vector<int>
    >& stream_task_order,

    std::unordered_map<
        int,
        tf::Task
    >& task_nodes
) {
  std::unordered_map<
      int,
      const TaskSpec*
  > task_by_id;


  task_by_id.reserve(
      tasks.size()
  );


  /*
   * STG依存
   */
  for (
      const auto& task :
      tasks
  ) {
    task_by_id.emplace(
        task.id,
        &task
    );


    for (
        const int pred_id :
        task.preds
    ) {
      const auto pred_it =
          task_nodes.find(
              pred_id
          );


      const auto current_it =
          task_nodes.find(
              task.id
          );


      if (
          pred_it ==
              task_nodes.end() ||

          current_it ==
              task_nodes.end()
      ) {
        throw std::runtime_error(
            "Taskflow dependency references "
            "an unknown task"
        );
      }


      pred_it->second.precede(
          current_it->second
      );
    }
  }


  /*
   * 同一Stream順序
   */
  for (
      const auto& task_ids :
      stream_task_order
  ) {
    for (
        std::size_t index = 1;
        index < task_ids.size();
        ++index
    ) {
      const int previous_id =
          task_ids.at(
              index - 1
          );


      const int current_id =
          task_ids.at(
              index
          );


      const TaskSpec& current_task =
          *task_by_id.at(
              current_id
          );


      const bool direct_dependency =
          std::find(
              current_task.preds.begin(),
              current_task.preds.end(),
              previous_id
          ) !=
          current_task.preds.end();


      if (
          !direct_dependency
      ) {
        task_nodes.at(
            previous_id
        ).precede(
            task_nodes.at(
                current_id
            )
        );
      }
    }
  }
}


}  // namespace stage5_detail


/*
 * ================================================================
 * Stage 5本体
 * ================================================================
 */
inline GreenContextExecutionResult
execute_with_green_context(
    const std::vector<TaskSpec>& tasks,

    const StreamScheduleResult& schedule,

    const std::vector<int>& stream_sm_counts,

    const GreenContextExecutionOptions& options = {}
) {
  /*
   * ============================================================
   * 入力確認
   * ============================================================
   */
  if (
      tasks.empty()
  ) {
    throw std::invalid_argument(
        "tasks must not be empty"
    );
  }


  validate_stream_sm_counts(
      stream_sm_counts
  );


  if (
      options.background_chunk_count <= 0
  ) {
    throw std::invalid_argument(
        "background_chunk_count must be positive"
    );
  }


  if (
      options.scheduling_reference_sm <= 0
  ) {
    throw std::invalid_argument(
        "scheduling_reference_sm must be positive"
    );
  }


  const int stream_count =
      static_cast<int>(
          stream_sm_counts.size()
      );


  validate_execution_schedule(
      tasks,
      schedule,
      stream_count
  );


  /*
   * ============================================================
   * Stage 3のStream順序
   * ============================================================
   */
  const auto stream_task_order =
      stage5_detail::
          make_stream_task_order(
              tasks,
              schedule,
              stream_count
          );


  const auto last_task_of_stream =
      stage5_detail::
          make_last_task_of_stream(
              stream_task_order
          );


  /*
   * ============================================================
   * GPU memory
   * ============================================================
   */
  constexpr int element_count =
      kTaskElementCount;


  const std::size_t allocation_element_count =
      static_cast<std::size_t>(
          element_count
      ) *
      static_cast<std::size_t>(
          stream_count
      );


  float* device_memory =
      nullptr;


  GC_CUDA_CHECK(
      cudaMalloc(
          &device_memory,
          allocation_element_count *
              sizeof(float)
      )
  );


  GC_CUDA_CHECK(
      cudaMemset(
          device_memory,
          0,
          allocation_element_count *
              sizeof(float)
      )
  );


  GreenContextExecutionResult result;


  RuntimeResources resources;


  std::mutex resource_mutex;


  /*
   * ============================================================
   * CUDA Resource作成
   * ============================================================
   */
  const auto setup_start =
      std::chrono::steady_clock::now();


  if (
      options.disable_green_context
  ) {
    resources =
        create_normal_stream_resources(
            stream_count,
            options.scheduling_reference_sm
        );
  }

  else {
    resources =
        create_green_context_resources(
            stream_sm_counts
        );
  }


  const auto setup_end =
      std::chrono::steady_clock::now();


  result.resource_setup_ms =
      std::chrono::duration<
          double,
          std::milli
      >(
          setup_end -
          setup_start
      ).count();


  print_runtime_resource_state(
      resources,
      "initial"
  );


  /*
   * ============================================================
   * タスクがないGCは即解放
   * ============================================================
   */
  if (
      !options.disable_green_context
  ) {
    for (
        int stream_id = 1;
        stream_id < stream_count;
        ++stream_id
    ) {
      if (
          last_task_of_stream.at(
              static_cast<std::size_t>(
                  stream_id
              )
          ) ==
          -1
      ) {
        release_green_context_stream(
            resources,
            stream_id,
            resource_mutex
        );
      }
    }
  }


  /*
   * ============================================================
   * Taskflow
   * ============================================================
   */
  tf::Executor executor(
      8
  );


  tf::Taskflow taskflow;


  std::unordered_map<
      int,
      tf::Task
  > task_nodes;


  std::unordered_map<
      int,
      cudaEvent_t
  > done_events;


  std::unordered_map<
      int,
      cudaEvent_t
  > kernel_start_events;


  std::unordered_map<
      int,
      cudaEvent_t
  > kernel_stop_events;


  task_nodes.reserve(
      tasks.size()
  );


  done_events.reserve(
      tasks.size()
  );


  kernel_start_events.reserve(
      tasks.size()
  );


  kernel_stop_events.reserve(
      tasks.size()
  );


  /*
   * ============================================================
   * GPU時間測定基準
   * ============================================================
   */
  cudaStream_t timing_stream =
      nullptr;


  cudaEvent_t origin_event =
      nullptr;


  GC_CUDA_CHECK(
      cudaStreamCreateWithFlags(
          &timing_stream,
          cudaStreamNonBlocking
      )
  );


  GC_CUDA_CHECK(
      cudaEventCreate(
          &origin_event
      )
  );


  GC_CUDA_CHECK(
      cudaEventRecord(
          origin_event,
          timing_stream
      )
  );


  GC_CUDA_CHECK(
      cudaEventSynchronize(
          origin_event
      )
  );


  /*
   * ============================================================
   * Event作成
   * ============================================================
   */
  for (
      const auto& task :
      tasks
  ) {
    cudaEvent_t done_event =
        nullptr;


    cudaEvent_t start_event =
        nullptr;


    cudaEvent_t stop_event =
        nullptr;


    GC_CUDA_CHECK(
        cudaEventCreateWithFlags(
            &done_event,
            cudaEventDisableTiming
        )
    );


    GC_CUDA_CHECK(
        cudaEventCreate(
            &start_event
        )
    );


    GC_CUDA_CHECK(
        cudaEventCreate(
            &stop_event
        )
    );


    done_events.emplace(
        task.id,
        done_event
    );


    kernel_start_events.emplace(
        task.id,
        start_event
    );


    kernel_stop_events.emplace(
        task.id,
        stop_event
    );
  }


  /*
   * ============================================================
   * Taskflow Node
   * ============================================================
   */
  for (
      const auto& task :
      tasks
  ) {
    tf::Task node =
        taskflow.emplace(
            [&, task]() {

              const int stream_id =
                  schedule.task_stream.at(
                      task.id
                  );


              cudaStream_t stream =
                  get_stream_by_id(
                      resources,
                      stream_id
                  );


              /*
               * ==============================================
               * STG依存待ち
               * ==============================================
               */
              for (
                  const int pred_id :
                  task.preds
              ) {
                const auto event_it =
                    done_events.find(
                        pred_id
                    );


                if (
                    event_it ==
                    done_events.end()
                ) {
                  throw std::runtime_error(
                      "completion event not found for predecessor " +
                      std::to_string(
                          pred_id
                      )
                  );
                }


                GC_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        event_it->second,
                        0
                    )
                );
              }


              GC_CUDA_CHECK(
                  cudaEventRecord(
                      kernel_start_events.at(
                          task.id
                      ),
                      stream
                  )
              );


              float* task_memory =
                  device_memory +
                  static_cast<std::size_t>(
                      stream_id
                  ) *
                  static_cast<std::size_t>(
                      element_count
                  );


              /*
               * ==============================================
               * 実際にこのStreamで利用可能なSM数
               * ==============================================
               */
              int configured_stream_sm =
                  options.scheduling_reference_sm;


              /*
               * GC使用時
               */
              if (
                  !options.disable_green_context
              ) {
                /*
                 * Stream 0はGC解放によってSM数が変化する。
                 */
                if (
                    stream_id == 0
                ) {
                  configured_stream_sm =
                      get_current_normal_sm_count(
                          resources,
                          resource_mutex
                      );
                }

                /*
                 * GC側はStage 4で決定した固定SM数。
                 */
                else {
                  configured_stream_sm =
                      stream_sm_counts.at(
                          static_cast<std::size_t>(
                              stream_id
                          )
                      );
                }
              }


              /*
               * ==============================================
               * Kernel起動
               * ==============================================
               */
              if (
                  options.disable_green_context
              ) {
                const int chunk_count =
                    stream_id == 0
                        ? 1
                        : options.background_chunk_count;


                stage5_detail::
                    launch_task_kernel_chunked(
                        task,
                        task_memory,
                        element_count,
                        stream,
                        chunk_count,
                        configured_stream_sm
                    );
              }

              else {
                stage5_detail::
                    launch_configured_task_kernel(
                        task,
                        task_memory,
                        element_count,
                        stream,
                        configured_stream_sm
                    );
              }


              /*
               * Stream 0確認ログ
               */
              if (
                  !options.disable_green_context &&
                  stream_id == 0
              ) {
                const int grid_blocks =
                    stage5_detail::
                        calculate_grid_block_count(
                            task,
                            configured_stream_sm
                        );


                const int threads_per_block =
                    stage5_detail::
                        calculate_threads_per_block(
                            element_count,
                            grid_blocks
                        );


                std::cout
                    << "[normal launch]"
                    << " task="
                    << task.id

                    << ", SM="
                    << configured_stream_sm

                    << ", blocks="
                    << grid_blocks

                    << ", threads/block="
                    << threads_per_block

                    << '\n';
              }


              GC_CUDA_CHECK(
                  cudaEventRecord(
                      kernel_stop_events.at(
                          task.id
                      ),
                      stream
                  )
              );


              GC_CUDA_CHECK(
                  cudaEventRecord(
                      done_events.at(
                          task.id
                      ),
                      stream
                  )
              );


              /*
               * ==============================================
               * Stream 0
               *
               * 次タスクを先行投入しない。
               *
               * GCがこのタスク実行中に解放された場合、
               * 次のタスクは増加後のSM数で起動できる。
               * ==============================================
               */
              if (
                  !options.disable_green_context &&
                  stream_id == 0
              ) {
                GC_CUDA_CHECK(
                    cudaEventSynchronize(
                        done_events.at(
                            task.id
                        )
                    )
                );
              }


              /*
               * ==============================================
               * GC Streamの最後のタスク
               * ==============================================
               */
              if (
                  !options.disable_green_context &&

                  stream_id > 0 &&

                  task.id ==
                      last_task_of_stream.at(
                          static_cast<std::size_t>(
                              stream_id
                          )
                      )
              ) {
                /*
                 * GPU上で本当に完了したことを確認。
                 */
                GC_CUDA_CHECK(
                    cudaEventSynchronize(
                        done_events.at(
                            task.id
                        )
                    )
                );


                std::cout
                    << "[GC last task completed]"
                    << " stream="
                    << stream_id

                    << ", task="
                    << task.id

                    << '\n';


                /*
                 * 即GC解放。
                 */
                release_green_context_stream(
                    resources,
                    stream_id,
                    resource_mutex
                );
              }
            }
        );


    node.name(
        std::to_string(
            task.id
        )
    );


    task_nodes.emplace(
        task.id,
        node
    );
  }


  /*
   * ============================================================
   * STG依存 + Stream順序
   * ============================================================
   */
  stage5_detail::
      add_taskflow_dependencies(
          tasks,
          stream_task_order,
          task_nodes
      );


  /*
   * ============================================================
   * GPU実行
   * ============================================================
   */
  const auto submit_start =
      std::chrono::steady_clock::now();


  executor.run(
      taskflow
  ).wait();


  /*
   * 残っているGCがあればcleanup。
   */
  if (
      !options.disable_green_context &&
      stream_count >= 2
  ) {
    wait_and_release_green_context_streams(
        resources,
        resource_mutex
    );
  }


  GC_CUDA_CHECK(
      cudaDeviceSynchronize()
  );


  const auto submit_end =
      std::chrono::steady_clock::now();


  result.gpu_submit_wait_ms =
      std::chrono::duration<
          double,
          std::milli
      >(
          submit_end -
          submit_start
      ).count();


  /*
   * ============================================================
   * GPU kernel区間
   * ============================================================
   */
  double first_kernel_start_ms =
      std::numeric_limits<double>::max();


  double last_kernel_stop_ms =
      0.0;


  for (
      const auto& task :
      tasks
  ) {
    float start_ms =
        0.0f;


    float stop_ms =
        0.0f;


    GC_CUDA_CHECK(
        cudaEventElapsedTime(
            &start_ms,
            origin_event,
            kernel_start_events.at(
                task.id
            )
        )
    );


    GC_CUDA_CHECK(
        cudaEventElapsedTime(
            &stop_ms,
            origin_event,
            kernel_stop_events.at(
                task.id
            )
        )
    );


    first_kernel_start_ms =
        std::min(
            first_kernel_start_ms,
            static_cast<double>(
                start_ms
            )
        );


    last_kernel_stop_ms =
        std::max(
            last_kernel_stop_ms,
            static_cast<double>(
                stop_ms
            )
        );
  }


  result.gpu_kernel_ms =
      tasks.empty()
          ? 0.0
          :
            last_kernel_stop_ms -
            first_kernel_start_ms;


  /*
   * ============================================================
   * 最終状態
   * ============================================================
   */
  print_runtime_resource_state(
      resources,
      "after dynamic GC release"
  );


  if (
      !options.disable_green_context
  ) {
    if (
        all_green_context_streams_released(
            resources,
            resource_mutex
        )
    ) {
      std::cout
          << "[CHECK] all GC streams released\n";
    }

    else {
      std::cout
          << "[CHECK] some GC streams are still alive\n";
    }


    const int final_normal_sm =
        get_current_normal_sm_count(
            resources,
            resource_mutex
        );


    std::cout
        << "[CHECK] final normal SM="
        << final_normal_sm
        << '\n';
  }


  /*
   * ============================================================
   * Event解放
   * ============================================================
   */
  for (
      auto& [task_id, event] :
      done_events
  ) {
    (void)task_id;


    GC_CUDA_CHECK(
        cudaEventDestroy(
            event
        )
    );
  }


  for (
      auto& [task_id, event] :
      kernel_start_events
  ) {
    (void)task_id;


    GC_CUDA_CHECK(
        cudaEventDestroy(
            event
        )
    );
  }


  for (
      auto& [task_id, event] :
      kernel_stop_events
  ) {
    (void)task_id;


    GC_CUDA_CHECK(
        cudaEventDestroy(
            event
        )
    );
  }


  GC_CUDA_CHECK(
      cudaEventDestroy(
          origin_event
      )
  );


  GC_CUDA_CHECK(
      cudaStreamDestroy(
          timing_stream
      )
  );


  destroy_runtime_resources(
      resources
  );


  GC_CUDA_CHECK(
      cudaFree(
          device_memory
      )
  );


  return result;
}


/*
 * ================================================================
 * Stage 4結果を直接受け取るAPI
 * ================================================================
 */
inline GreenContextExecutionResult
execute_with_green_context(
    const std::vector<TaskSpec>& tasks,

    const SmAllocationDecision& decision,

    const GreenContextExecutionOptions& options = {}
) {
  if (
      decision.sm_counts.empty()
  ) {
    throw std::invalid_argument(
        "Stage 4 decision has no SM allocation"
    );
  }


  if (
      !std::isfinite(
          decision.estimated_makespan
      ) ||

      decision.estimated_makespan <
          0.0
  ) {
    throw std::invalid_argument(
        "Stage 4 decision has invalid estimated makespan"
    );
  }


  return
      execute_with_green_context(
          tasks,
          decision.schedule,
          decision.sm_counts,
          options
      );
}


#undef GC_CUDA_CHECK