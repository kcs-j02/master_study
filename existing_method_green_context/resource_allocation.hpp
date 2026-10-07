

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common_types.hpp"
#include "task_levelization.hpp"

#define RA_CUDA_CHECK(expr)                                                   \
  do {                                                                        \
    cudaError_t _err = (expr);                                                \
    if (_err != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                 \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      throw std::runtime_error(cudaGetErrorString(_err));                     \
    }                                                                         \
  } while (0)

struct RuntimeResources {
  std::vector<cudaStream_t> streams;
  std::vector<cudaExecutionContext_t> green_ctxs;
  std::vector<cudaDevResourceDesc_t> resource_descs;
  std::vector<int> sm_count_each_stream;
  std::vector<long long> priority_each_stream;

  // 比較条件では全streamをGreen Context上に作成する。
  int normal_stream_id = -1;
  std::vector<int> gc_stream_ids;
};

inline int align_up_int(int value, int alignment) {
  if (alignment <= 0) {
    throw std::invalid_argument("alignment must be positive");
  }

  return ((value + alignment - 1) / alignment) * alignment;
}

inline int align_down_int(int value, int alignment) {
  if (alignment <= 0) {
    throw std::invalid_argument("alignment must be positive");
  }

  return (value / alignment) * alignment;
}

inline cudaStream_t get_stream_by_id(
    const RuntimeResources& rr,
    int stream_id
) {
  if (stream_id < 0 ||
      stream_id >= static_cast<int>(rr.streams.size())) {
    throw std::runtime_error(
        "invalid stream_id: " + std::to_string(stream_id)
    );
  }

  const cudaStream_t stream =
      rr.streams.at(static_cast<std::size_t>(stream_id));

  if (stream == nullptr) {
    throw std::runtime_error(
        "stream is not available: " + std::to_string(stream_id)
    );
  }

  return stream;
}

inline bool is_gc_stream_id(
    const RuntimeResources& rr,
    int stream_id
) {
  return std::find(
      rr.gc_stream_ids.begin(),
      rr.gc_stream_ids.end(),
      stream_id
  ) != rr.gc_stream_ids.end();
}

struct SmPartitionInfo {
  int available_sm = 0;
  int unit_sm = 0;
  int min_group_sm = 0;
};

inline SmPartitionInfo query_sm_partition_info(
    int gpu_device_index = 0
) {
  RA_CUDA_CHECK(cudaSetDevice(gpu_device_index));

  cudaDevResource initial_sm_resource {};

  RA_CUDA_CHECK(
      cudaDeviceGetDevResource(
          gpu_device_index,
          &initial_sm_resource,
          cudaDevResourceTypeSm
      )
  );

  const int available_sm =
      static_cast<int>(initial_sm_resource.sm.smCount);

  const int min_partition_size =
      static_cast<int>(
          initial_sm_resource.sm.minSmPartitionSize
      );

  const int coscheduled_alignment =
      static_cast<int>(
          initial_sm_resource.sm.smCoscheduledAlignment
      );

  const int unit_sm =
      std::max(2, coscheduled_alignment);

  const int min_group_sm =
      align_up_int(
          std::max(2, min_partition_size),
          unit_sm
      );

  if (available_sm <= 0 ||
      unit_sm <= 0 ||
      min_group_sm <= 0) {
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
 * 使用可能なSMを、分割粒度を守りながら全GC streamへ均等配分する。
 * 粒度未満の端数SMは使用せず、余った配分単位はstream IDの小さい
 * GCへ1単位ずつ割り当てる。
 *
 * タスク重要度による重み付けは行わない。
 * 比較条件「既存手法 + 均等Green Context」専用の配分である。
 */
inline std::vector<int> make_even_gc_sm_counts(
    int stream_count,
    int gpu_device_index = 0
) {
  if (stream_count <= 0) {
    throw std::invalid_argument(
        "stream_count must be positive"
    );
  }

  const SmPartitionInfo info =
      query_sm_partition_info(gpu_device_index);

  const int total_units =
      info.available_sm / info.unit_sm;

  const int minimum_units =
      info.min_group_sm / info.unit_sm;

  if (total_units < stream_count * minimum_units) {
    throw std::runtime_error(
        "not enough SMs for evenly divided Green Contexts"
    );
  }

  const int base_units = total_units / stream_count;
  const int remainder_units = total_units % stream_count;

  std::vector<int> sm_counts(
      static_cast<std::size_t>(stream_count),
      0
  );

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    const int units =
        base_units + (stream_id < remainder_units ? 1 : 0);

    sm_counts.at(static_cast<std::size_t>(stream_id)) =
        units * info.unit_sm;
  }

  const int allocated_sm =
      std::accumulate(sm_counts.begin(), sm_counts.end(), 0);

  std::cout
      << "===== Existing method + even GC partition =====\n"
      << "available SM : " << info.available_sm << '\n'
      << "unit SM      : " << info.unit_sm << '\n'
      << "allocated SM : " << allocated_sm << '\n'
      << "unused SM    : " << info.available_sm - allocated_sm
      << '\n';

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    std::cout
        << "stream[" << stream_id << "] SM="
        << sm_counts.at(static_cast<std::size_t>(stream_id))
        << " [GC]\n";
  }

  std::cout << "================================================\n";

  return sm_counts;
}

/*
 * 通常streamへ優先的にSMを残すときのGC総SM数を求める。
 *
 * normal_stream_percent=70、available_sm=114、unit_sm=8なら、
 * GC側はfloor(114 * 30% / 8) * 8 = 32 SM、
 * 通常stream側は114 - 32 = 82 SMになる。
 */
inline int calculate_gc_sm_budget(
    const SmPartitionInfo& info,
    int gc_stream_count,
    int normal_stream_percent = 70
) {
  if (gc_stream_count <= 0) {
    throw std::invalid_argument(
        "gc_stream_count must be positive"
    );
  }

  if (normal_stream_percent <= 0 ||
      normal_stream_percent >= 100) {
    throw std::invalid_argument(
        "normal_stream_percent must be between 1 and 99"
    );
  }

  const int minimum_gc_sm =
      gc_stream_count * info.min_group_sm;

  const int target_gc_sm =
      info.available_sm *
      (100 - normal_stream_percent) /
      100;

  int gc_sm_budget =
      align_down_int(target_gc_sm, info.unit_sm);

  gc_sm_budget = std::max(
      gc_sm_budget,
      minimum_gc_sm
  );

  if (gc_sm_budget >= info.available_sm) {
    throw std::runtime_error(
        "not enough SMs left for the normal stream"
    );
  }

  return gc_sm_budget;
}

/*
 * 通常stream優先のSM配分で作成可能な最大stream数。
 * stream 0が通常stream、stream 1以降がGC。
 */
inline int get_normal_priority_stream_limit(
    const SmPartitionInfo& info,
    int normal_stream_percent = 70
) {
  if (normal_stream_percent <= 0 ||
      normal_stream_percent >= 100) {
    throw std::invalid_argument(
        "normal_stream_percent must be between 1 and 99"
    );
  }

  const int target_gc_sm =
      align_down_int(
          info.available_sm *
          (100 - normal_stream_percent) /
          100,
          info.unit_sm
      );

  return 1 + target_gc_sm / info.min_group_sm;
}

/*
 * stream 0を通常streamとして残し、stream 1以降をGCにする。
 *
 * 方針:
 *   - 通常streamへ既定で70%程度のSMを残す。
 *   - GC総量だけをcudaDevSmResourceSplitで確保する。
 *   - GC間は極端な性能差を付けず、ほぼ均等に分配する。
 *   - 重要タスクはtask_assignment側のbottom level順に処理され、
 *     最大SMのstream 0が選ばれやすくなる。
 */
inline std::vector<int> decide_stream_sm_counts(
    const std::vector<TaskSpec>& tasks,
    const TaskLevels& levels,
    int stream_count,
    int gpu_device_index = 0,
    int normal_stream_percent = 70
) {
  if (stream_count < 2) {
    throw std::invalid_argument(
        "stream_count must be at least 2"
    );
  }

  if (tasks.empty()) {
    throw std::invalid_argument(
        "tasks must not be empty"
    );
  }

  if (levels.empty()) {
    throw std::invalid_argument(
        "levels must not be empty"
    );
  }

  const SmPartitionInfo info =
      query_sm_partition_info(gpu_device_index);

  const int normal_stream_id = 0;
  const int gc_stream_count = stream_count - 1;

  const int gc_sm_budget =
      calculate_gc_sm_budget(
          info,
          gc_stream_count,
          normal_stream_percent
      );

  const int gc_units =
      gc_sm_budget / info.unit_sm;

  const int minimum_units =
      info.min_group_sm / info.unit_sm;

  if (gc_units < gc_stream_count * minimum_units) {
    throw std::runtime_error(
        "insufficient GC SM budget for requested stream count"
    );
  }

  std::vector<int> sm_counts(
      static_cast<std::size_t>(stream_count),
      0
  );

  const int base_units =
      gc_units / gc_stream_count;

  const int remainder_units =
      gc_units % gc_stream_count;

  for (int gc_index = 0;
       gc_index < gc_stream_count;
       ++gc_index) {
    const int stream_id = gc_index + 1;

    const int units =
        base_units +
        (gc_index < remainder_units ? 1 : 0);

    const int sm_count = units * info.unit_sm;

    if (sm_count < info.min_group_sm) {
      throw std::runtime_error(
          "calculated GC SM count is below the minimum"
      );
    }

    sm_counts.at(
        static_cast<std::size_t>(stream_id)
    ) = sm_count;
  }

  const int total_gc_sm =
      std::accumulate(
          sm_counts.begin() + 1,
          sm_counts.end(),
          0
      );

  sm_counts.at(
      static_cast<std::size_t>(normal_stream_id)
  ) = info.available_sm - total_gc_sm;

  if (sm_counts.at(0) <= 0) {
    throw std::runtime_error(
        "normal stream has no remaining SMs"
    );
  }

  std::cout
      << "===== Planned SM partition =====\n"
      << "available SM       : "
      << info.available_sm << '\n'
      << "unit SM            : "
      << info.unit_sm << '\n'
      << "minimum GC SM      : "
      << info.min_group_sm << '\n'
      << "normal target      : "
      << normal_stream_percent << "%\n"
      << "normal stream id   : "
      << normal_stream_id << '\n'
      << "normal estimated SM: "
      << sm_counts.at(0) << '\n'
      << "total GC SM        : "
      << total_gc_sm << '\n';

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    std::cout
        << "stream[" << stream_id << "] SM="
        << sm_counts.at(
               static_cast<std::size_t>(stream_id)
           );

    if (stream_id == normal_stream_id) {
      std::cout << " [normal]";
    } else {
      std::cout << " [GC]";
    }

    std::cout << '\n';
  }

  std::cout
      << "================================\n";

  return sm_counts;
}

/*
 * 全streamを、それぞれ均等配分されたSMを持つGreen Context上に作る。
 */
inline RuntimeResources create_runtime_resources(
    const std::vector<int>& stream_sm_counts,
    int gpu_device_index = 0
) {
  if (stream_sm_counts.empty()) {
    throw std::invalid_argument(
        "at least one stream is required"
    );
  }

  RA_CUDA_CHECK(cudaSetDevice(gpu_device_index));

  const SmPartitionInfo info =
      query_sm_partition_info(gpu_device_index);

  const int stream_count =
      static_cast<int>(stream_sm_counts.size());

  const int gc_stream_count = stream_count;

  int requested_gc_sm = 0;

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    const int sm_count =
        stream_sm_counts.at(
            static_cast<std::size_t>(stream_id)
        );

    if (sm_count < info.min_group_sm) {
      throw std::invalid_argument(
          "GC SM count is smaller than minimum: stream " +
          std::to_string(stream_id)
      );
    }

    if (sm_count % info.unit_sm != 0) {
      throw std::invalid_argument(
          "GC SM count is not aligned: stream " +
          std::to_string(stream_id)
      );
    }

    requested_gc_sm += sm_count;
  }

  if (requested_gc_sm > info.available_sm) {
    throw std::runtime_error(
        "Green Context SM request exceeds available SMs"
    );
  }

  RuntimeResources rr;

  rr.streams.resize(
      static_cast<std::size_t>(stream_count),
      nullptr
  );

  rr.green_ctxs.resize(
      static_cast<std::size_t>(stream_count),
      nullptr
  );

  rr.resource_descs.resize(
      static_cast<std::size_t>(stream_count),
      nullptr
  );

  rr.sm_count_each_stream = stream_sm_counts;

  rr.priority_each_stream.resize(
      static_cast<std::size_t>(stream_count),
      0
  );

  rr.normal_stream_id = -1;

  rr.gc_stream_ids.reserve(
      static_cast<std::size_t>(gc_stream_count)
  );

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    rr.gc_stream_ids.push_back(stream_id);
  }

  cudaDevResource initial_sm_resource {};

  RA_CUDA_CHECK(
      cudaDeviceGetDevResource(
          gpu_device_index,
          &initial_sm_resource,
          cudaDevResourceTypeSm
      )
  );

  std::vector<cudaDevResource> split_sm_resources(
      static_cast<std::size_t>(gc_stream_count)
  );

  std::vector<cudaDevSmResourceGroupParams> group_params(
      static_cast<std::size_t>(gc_stream_count)
  );

  for (int gc_index = 0;
       gc_index < gc_stream_count;
       ++gc_index) {
    const int stream_id = gc_index;

    auto& params =
        group_params.at(
            static_cast<std::size_t>(gc_index)
        );

    params = {};

    params.smCount =
        static_cast<unsigned int>(
            stream_sm_counts.at(
                static_cast<std::size_t>(stream_id)
            )
        );

    // GH100で過度な8-SM co-scheduling制約を避ける。
    params.coscheduledSmCount = 2;
    params.preferredCoscheduledSmCount = 2;
    params.flags = 0;
  }

  RA_CUDA_CHECK(cudaDeviceSynchronize());

  const auto setup_start =
      std::chrono::high_resolution_clock::now();

  const cudaError_t split_error =
      cudaDevSmResourceSplit(
          split_sm_resources.data(),
          static_cast<unsigned int>(gc_stream_count),
          &initial_sm_resource,
          nullptr,
          0,
          group_params.data()
      );

  if (split_error != cudaSuccess) {
    std::cerr
        << "cudaDevSmResourceSplit failed\n"
        << "available_sm=" << info.available_sm
        << ", requested_gc_sm=" << requested_gc_sm
        << ", gc_stream_count=" << gc_stream_count
        << '\n';

    for (int gc_index = 0;
         gc_index < gc_stream_count;
         ++gc_index) {
      std::cerr
          << "  group[" << gc_index << "]"
          << " smCount="
          << group_params.at(
                 static_cast<std::size_t>(gc_index)
             ).smCount
          << " coscheduledSmCount="
          << group_params.at(
                 static_cast<std::size_t>(gc_index)
             ).coscheduledSmCount
          << '\n';
    }

    RA_CUDA_CHECK(split_error);
  }

  for (int gc_index = 0;
       gc_index < gc_stream_count;
       ++gc_index) {
    const int stream_id = gc_index;

    RA_CUDA_CHECK(
        cudaDevResourceGenerateDesc(
            &rr.resource_descs.at(
                static_cast<std::size_t>(stream_id)
            ),
            &split_sm_resources.at(
                static_cast<std::size_t>(gc_index)
            ),
            1
        )
    );
  }

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    RA_CUDA_CHECK(
        cudaGreenCtxCreate(
            &rr.green_ctxs.at(
                static_cast<std::size_t>(stream_id)
            ),
            rr.resource_descs.at(
                static_cast<std::size_t>(stream_id)
            ),
            gpu_device_index,
            0
        )
    );
  }

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    RA_CUDA_CHECK(
        cudaExecutionCtxStreamCreate(
            &rr.streams.at(
                static_cast<std::size_t>(stream_id)
            ),
            rr.green_ctxs.at(
                static_cast<std::size_t>(stream_id)
            ),
            cudaStreamNonBlocking,
            0
        )
    );
  }

  const auto setup_stop =
      std::chrono::high_resolution_clock::now();

  const double setup_ms =
      std::chrono::duration<double, std::milli>(
          setup_stop - setup_start
      ).count();

  std::cout
      << "[GC setup time] "
      << setup_ms
      << " ms\n";

  return rr;
}

inline void print_runtime_resource_state(
    const RuntimeResources& rr,
    const std::string& tag
) {
  std::cout
      << "\n===== Runtime Resource State: "
      << tag
      << " =====\n";

  for (int stream_id = 0;
       stream_id < static_cast<int>(rr.streams.size());
       ++stream_id) {
    const bool has_stream =
        rr.streams.at(
            static_cast<std::size_t>(stream_id)
        ) != nullptr;

    const bool has_gc =
        rr.green_ctxs.at(
            static_cast<std::size_t>(stream_id)
        ) != nullptr;

    std::cout
        << "stream[" << stream_id << "] ";

    if (stream_id == rr.normal_stream_id) {
      std::cout << "[normal] ";
    } else if (is_gc_stream_id(rr, stream_id)) {
      std::cout << "[GC] ";
    } else {
      std::cout << "[unused] ";
    }

    std::cout
        << "stream_ptr="
        << rr.streams.at(
               static_cast<std::size_t>(stream_id)
           )
        << ", green_ctx="
        << rr.green_ctxs.at(
               static_cast<std::size_t>(stream_id)
           )
        << ", has_stream="
        << has_stream
        << ", has_gc="
        << has_gc
        << ", SM="
        << rr.sm_count_each_stream.at(
               static_cast<std::size_t>(stream_id)
           )
        << '\n';
  }

  std::cout
      << "=============================================\n";
}

inline void print_runtime_resource_state(
    const RuntimeResources& rr,
    int /* stream_count */,
    const std::string& tag
) {
  print_runtime_resource_state(rr, tag);
}

inline void release_gc_stream(
    RuntimeResources& rr,
    int stream_id
) {
  if (stream_id < 0 ||
      stream_id >= static_cast<int>(rr.streams.size())) {
    throw std::runtime_error("invalid stream index");
  }

  if (!is_gc_stream_id(rr, stream_id)) {
    return;
  }

  if (rr.green_ctxs.at(
          static_cast<std::size_t>(stream_id)
      ) == nullptr) {
    return;
  }

  if (rr.streams.at(
          static_cast<std::size_t>(stream_id)
      ) != nullptr) {
    RA_CUDA_CHECK(
        cudaStreamSynchronize(
            rr.streams.at(
                static_cast<std::size_t>(stream_id)
            )
        )
    );

    RA_CUDA_CHECK(
        cudaStreamDestroy(
            rr.streams.at(
                static_cast<std::size_t>(stream_id)
            )
        )
    );

    rr.streams.at(
        static_cast<std::size_t>(stream_id)
    ) = nullptr;
  }

  RA_CUDA_CHECK(
      cudaExecutionCtxDestroy(
          rr.green_ctxs.at(
              static_cast<std::size_t>(stream_id)
          )
      )
  );

  rr.green_ctxs.at(
      static_cast<std::size_t>(stream_id)
  ) = nullptr;

  rr.sm_count_each_stream.at(
      static_cast<std::size_t>(stream_id)
  ) = 0;

  std::cout
      << "[GC released] stream["
      << stream_id
      << "]\n";
}

inline void release_finished_gc_streams(
    RuntimeResources& rr
) {
  for (const int stream_id : rr.gc_stream_ids) {
    release_gc_stream(rr, stream_id);
  }
}

inline void release_finished_gc_streams(
    RuntimeResources& rr,
    int /* stream_count */
) {
  release_finished_gc_streams(rr);
}

inline bool all_gc_streams_released(
    const RuntimeResources& rr
) {
  for (const int stream_id : rr.gc_stream_ids) {
    if (rr.streams.at(
            static_cast<std::size_t>(stream_id)
        ) != nullptr) {
      return false;
    }

    if (rr.green_ctxs.at(
            static_cast<std::size_t>(stream_id)
        ) != nullptr) {
      return false;
    }

    if (rr.sm_count_each_stream.at(
            static_cast<std::size_t>(stream_id)
        ) != 0) {
      return false;
    }
  }

  return true;
}

inline bool all_gc_streams_released(
    const RuntimeResources& rr,
    int /* stream_count */
) {
  return all_gc_streams_released(rr);
}

inline void destroy_runtime_resources(
    RuntimeResources& rr
) {
  RA_CUDA_CHECK(cudaDeviceSynchronize());

  for (auto& stream : rr.streams) {
    if (stream != nullptr) {
      RA_CUDA_CHECK(cudaStreamDestroy(stream));
      stream = nullptr;
    }
  }

  for (auto& context : rr.green_ctxs) {
    if (context != nullptr) {
      RA_CUDA_CHECK(cudaExecutionCtxDestroy(context));
      context = nullptr;
    }
  }

  rr.streams.clear();
  rr.green_ctxs.clear();
  rr.resource_descs.clear();
  rr.sm_count_each_stream.clear();
  rr.priority_each_stream.clear();
  rr.gc_stream_ids.clear();
  rr.normal_stream_id = -1;
}
