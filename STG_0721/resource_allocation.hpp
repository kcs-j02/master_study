#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#include "common_types.hpp"

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

  int normal_stream_id = -1;
  std::vector<int> gc_stream_ids;
};

inline int align_up_int(int x, int a) {
  return ((x + a - 1) / a) * a;
}

inline cudaStream_t get_stream_by_id(const RuntimeResources& rr, int stream_id) {
  if (stream_id < 0 || stream_id >= static_cast<int>(rr.streams.size())) {
    throw std::runtime_error("invalid stream_id: " + std::to_string(stream_id));
  }

  return rr.streams[stream_id];
}

inline bool is_gc_stream_id(const RuntimeResources& rr, int stream_id) {
  return std::find(
      rr.gc_stream_ids.begin(),
      rr.gc_stream_ids.end(),
      stream_id
  ) != rr.gc_stream_ids.end();
}

inline void check_resource(
    int available_sm,
    int reserve_sm,
    int normal_reserved_sm,
    int allocatable_sm,
    int unit,
    int min_group_sm,
    int stream_count,
    int gc_stream_count,
    int normal_stream_id,
    int remaining_sm,
    int remaining_chunks,
    int leftover_sm,
    const RuntimeResources& rr
) {
  std::cout << "===== Green Context Resource Allocation =====\n";

  std::cout << "available SM        = " << available_sm << "\n";
  std::cout << "base reserve SM     = " << reserve_sm << "\n";
  std::cout << "normal reserved SM  = " << normal_reserved_sm << "\n";
  std::cout << "allocatable SM      = " << allocatable_sm << "\n";
  std::cout << "unit SM             = " << unit << "\n";
  std::cout << "min group SM        = " << min_group_sm << "\n";
  std::cout << "stream count        = " << stream_count << "\n";
  std::cout << "GC stream count     = " << gc_stream_count << "\n";
  std::cout << "normal stream id    = " << normal_stream_id << "\n";
  std::cout << "remaining SM        = " << remaining_sm << "\n";
  std::cout << "remaining chunks    = " << remaining_chunks << "\n";
  std::cout << "leftover SM         = " << leftover_sm << "\n";

  int total_gc_sm = 0;

  for (int i = 0; i < stream_count; ++i) {
    bool is_gc = is_gc_stream_id(rr, i);

    std::cout << "stream[" << i << "] priority = "
              << rr.priority_each_stream[i]
              << ", SM = "
              << rr.sm_count_each_stream[i];

    if (is_gc) {
      std::cout << "  [GC]";
      total_gc_sm += rr.sm_count_each_stream[i];
    } else if (i == normal_stream_id) {
      std::cout << "  [normal / display only]";
    } else {
      std::cout << "  [unused]";
    }

    std::cout << "\n";
  }

  std::cout << "total GC SM         = " << total_gc_sm << "\n";
  std::cout << "total displayed SM  = "
            << total_gc_sm + normal_reserved_sm << "\n";

  std::cout << "=============================================\n";
}

inline RuntimeResources resource_allocation(
    const TaskDFG& dfg,
    const std::vector<TaskSpec>& tasks,
    int stream_count,
    int gpu_device_index = 0,
    int reserve_sm = 8
) {
  if (stream_count < 2) {
    throw std::runtime_error("stream_count must be at least 2");
  }

  RA_CUDA_CHECK(cudaSetDevice(gpu_device_index));

  RuntimeResources rr;
  rr.streams.resize(stream_count, nullptr);
  rr.green_ctxs.resize(stream_count, nullptr);
  rr.resource_descs.resize(stream_count, nullptr);
  rr.sm_count_each_stream.resize(stream_count, 0);
  rr.priority_each_stream.resize(stream_count, 0);

  // task_id -> order_count
  std::unordered_map<int, long long> order_count_by_task;

  for (const auto& t : tasks) {
    long long order_count = std::max(1, t.work_units);

    if (t.kind == KernelKind::HEAVY) {
      order_count *= 3;
    }

    order_count_by_task[t.id] = order_count;
  }

  // streamごとのpriorityをproc_timeで作る
  for (const auto& [task_id, ni] : dfg.nodes) {
    if (ni.stream_id >= 0 && ni.stream_id < stream_count) {
      rr.priority_each_stream[ni.stream_id] += std::max(1LL, ni.proc_time);
    }
  }

  long long total_priority = 0;
  for (long long p : rr.priority_each_stream) {
    total_priority += p;
  }

  if (total_priority == 0) {
    throw std::runtime_error("total priority is zero");
  }

  // priority最大のstreamをGCなし通常streamにする
  int normal_stream_id = 0;

  for (int i = 1; i < stream_count; ++i) {
    if (rr.priority_each_stream[i] > rr.priority_each_stream[normal_stream_id]) {
      normal_stream_id = i;
    }
  }

  std::vector<int> gc_stream_ids;
  gc_stream_ids.reserve(stream_count - 1);

  for (int i = 0; i < stream_count; ++i) {
    if (i != normal_stream_id) {
      gc_stream_ids.push_back(i);
    }
  }

  int gc_stream_count = static_cast<int>(gc_stream_ids.size());

  rr.normal_stream_id = normal_stream_id;
  rr.gc_stream_ids = gc_stream_ids;

  std::cout << "[normal stream selected] stream["
            << normal_stream_id
            << "] priority = "
            << rr.priority_each_stream[normal_stream_id]
            << "\n";

  // GPUのSM資源情報を取得
  cudaDevResource initial_sm_resource {};
  RA_CUDA_CHECK(cudaDeviceGetDevResource(
      gpu_device_index,
      &initial_sm_resource,
      cudaDevResourceTypeSm
  ));

  int available_sm = static_cast<int>(initial_sm_resource.sm.smCount);
  int min_part = static_cast<int>(initial_sm_resource.sm.minSmPartitionSize);
  int align = static_cast<int>(initial_sm_resource.sm.smCoscheduledAlignment);

  int unit = std::max(2, align);
  int min_group_sm = align_up_int(std::max(2, min_part), unit);

  if (available_sm - reserve_sm < gc_stream_count * min_group_sm) {
    reserve_sm = 0;
  }

  int allocatable_sm = available_sm - reserve_sm;

  if (allocatable_sm < gc_stream_count * min_group_sm) {
    throw std::runtime_error(
        "not enough SMs for Green Contexts: available_sm=" +
        std::to_string(available_sm) +
        ", gc_stream_count=" +
        std::to_string(gc_stream_count)
    );
  }

  // GC streamだけに最低SM数を割り当てる
  for (int sid : gc_stream_ids) {
    rr.sm_count_each_stream[sid] = min_group_sm;
  }

  // 通常streamはGCではないので、SM数は表示用
  rr.sm_count_each_stream[normal_stream_id] = 0;

  int remaining_sm = allocatable_sm - gc_stream_count * min_group_sm;
  int remaining_chunks = remaining_sm / unit;
  int leftover_sm = remaining_sm % unit;

  // GCに割り当てないSM数
  int normal_reserved_sm = reserve_sm + leftover_sm;

  // 残りSMをGC streamだけに配分
  for (int c = 0; c < remaining_chunks; ++c) {
    long double best_score = -1.0;
    int best_stream = gc_stream_ids[0];

    for (int sid : gc_stream_ids) {
      long double score =
          static_cast<long double>(rr.priority_each_stream[sid]) /
          static_cast<long double>(rr.sm_count_each_stream[sid]);

      if (score > best_score) {
        best_score = score;
        best_stream = sid;
      }
    }

    rr.sm_count_each_stream[best_stream] += unit;
  }

  // 通常stream側は表示用
  rr.sm_count_each_stream[normal_stream_id] = normal_reserved_sm;

  check_resource(
      available_sm,
      reserve_sm,
      normal_reserved_sm,
      allocatable_sm,
      unit,
      min_group_sm,
      stream_count,
      gc_stream_count,
      normal_stream_id,
      remaining_sm,
      remaining_chunks,
      leftover_sm,
      rr
  );

  std::vector<cudaDevResource> split_sm_resources(gc_stream_count);
  std::vector<cudaDevSmResourceGroupParams> group_params(gc_stream_count);

  for (int g = 0; g < gc_stream_count; ++g) {
    int sid = gc_stream_ids[g];

    group_params[g] = {};
    group_params[g].smCount =
        static_cast<unsigned int>(rr.sm_count_each_stream[sid]);
    group_params[g].coscheduledSmCount = 0;
    group_params[g].preferredCoscheduledSmCount = 0;
    group_params[g].flags = 0;
  }

  RA_CUDA_CHECK(cudaDeviceSynchronize());

  auto t0 = std::chrono::high_resolution_clock::now();

  RA_CUDA_CHECK(cudaDevSmResourceSplit(
      split_sm_resources.data(),
      static_cast<unsigned int>(gc_stream_count),
      &initial_sm_resource,
      nullptr,
      0,
      group_params.data()
  ));

  for (int g = 0; g < gc_stream_count; ++g) {
    int sid = gc_stream_ids[g];

    RA_CUDA_CHECK(cudaDevResourceGenerateDesc(
        &rr.resource_descs[sid],
        &split_sm_resources[g],
        1
    ));
  }

  for (int g = 0; g < gc_stream_count; ++g) {
    int sid = gc_stream_ids[g];

    RA_CUDA_CHECK(cudaGreenCtxCreate(
        &rr.green_ctxs[sid],
        rr.resource_descs[sid],
        gpu_device_index,
        0
    ));
  }

  for (int g = 0; g < gc_stream_count; ++g) {
    int sid = gc_stream_ids[g];

    RA_CUDA_CHECK(cudaExecutionCtxStreamCreate(
        &rr.streams[sid],
        rr.green_ctxs[sid],
        cudaStreamDefault,
        0
    ));
  }

  // priority最大のstreamだけGCなし通常stream
  RA_CUDA_CHECK(cudaStreamCreateWithFlags(
      &rr.streams[normal_stream_id],
      cudaStreamNonBlocking
  ));

  auto t1 = std::chrono::high_resolution_clock::now();

  double gc_setup_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "[GC setup time] " << gc_setup_ms << " ms\n";

  return rr;
}

inline void release_gc_stream(RuntimeResources& rr, int i) {
  if (i < 0 || i >= static_cast<int>(rr.streams.size())) {
    throw std::runtime_error("invalid stream index");
  }

  if (rr.green_ctxs[i] == nullptr) {
    return;
  }

  if (rr.streams[i] != nullptr) {
    RA_CUDA_CHECK(cudaStreamSynchronize(rr.streams[i]));
    RA_CUDA_CHECK(cudaStreamDestroy(rr.streams[i]));
    rr.streams[i] = nullptr;
  }

  RA_CUDA_CHECK(cudaExecutionCtxDestroy(rr.green_ctxs[i]));
  rr.green_ctxs[i] = nullptr;

  rr.sm_count_each_stream[i] = 0;

  std::cout << "[GC released] stream[" << i << "]\n";
}

inline void release_finished_gc_streams(RuntimeResources& rr) {
  if (rr.gc_stream_ids.empty()) {
    return;
  }

  std::vector<cudaEvent_t> gc_done_events(rr.gc_stream_ids.size(), nullptr);
  std::vector<bool> gc_released(rr.gc_stream_ids.size(), false);

  for (int k = 0; k < static_cast<int>(rr.gc_stream_ids.size()); ++k) {
    int sid = rr.gc_stream_ids[k];

    if (rr.streams[sid] == nullptr || rr.green_ctxs[sid] == nullptr) {
      gc_released[k] = true;
      continue;
    }

    RA_CUDA_CHECK(cudaEventCreateWithFlags(
        &gc_done_events[k],
        cudaEventDisableTiming
    ));

    RA_CUDA_CHECK(cudaEventRecord(
        gc_done_events[k],
        rr.streams[sid]
    ));
  }

  bool all_gc_released = false;

  while (!all_gc_released) {
    all_gc_released = true;

    for (int k = 0; k < static_cast<int>(rr.gc_stream_ids.size()); ++k) {
      if (gc_released[k]) {
        continue;
      }

      cudaError_t st = cudaEventQuery(gc_done_events[k]);

      if (st == cudaSuccess) {
        int sid = rr.gc_stream_ids[k];

        release_gc_stream(rr, sid);

        RA_CUDA_CHECK(cudaEventDestroy(gc_done_events[k]));
        gc_done_events[k] = nullptr;

        gc_released[k] = true;
      } else if (st == cudaErrorNotReady) {
        all_gc_released = false;
      } else {
        RA_CUDA_CHECK(st);
      }
    }
  }
}

// 古い呼び方との互換用
inline void release_finished_gc_streams(
    RuntimeResources& rr,
    int /* stream_count */
) {
  release_finished_gc_streams(rr);
}

inline void print_runtime_resource_state(
    const RuntimeResources& rr,
    const std::string& tag
) {
  std::cout << "\n===== Runtime Resource State: " << tag << " =====\n";

  for (int i = 0; i < static_cast<int>(rr.streams.size()); ++i) {
    bool has_stream = rr.streams[i] != nullptr;
    bool has_gc = rr.green_ctxs[i] != nullptr;
    bool is_gc = is_gc_stream_id(rr, i);

    std::cout << "stream[" << i << "] ";

    if (is_gc) {
      std::cout << "[GC] ";
    } else if (i == rr.normal_stream_id) {
      std::cout << "[normal] ";
    } else {
      std::cout << "[unused] ";
    }

    std::cout << "stream_ptr = " << rr.streams[i]
              << ", green_ctx = " << rr.green_ctxs[i]
              << ", has_stream = " << has_stream
              << ", has_gc = " << has_gc
              << ", SM = " << rr.sm_count_each_stream[i]
              << "\n";
  }

  std::cout << "=============================================\n";
}

// 古い呼び方との互換用
inline void print_runtime_resource_state(
    const RuntimeResources& rr,
    int /* stream_count */,
    const std::string& tag
) {
  print_runtime_resource_state(rr, tag);
}

inline bool all_gc_streams_released(const RuntimeResources& rr) {
  for (int sid : rr.gc_stream_ids) {
    if (rr.streams[sid] != nullptr) {
      return false;
    }

    if (rr.green_ctxs[sid] != nullptr) {
      return false;
    }

    if (rr.sm_count_each_stream[sid] != 0) {
      return false;
    }
  }

  return true;
}

// 古い呼び方との互換用
inline bool all_gc_streams_released(
    const RuntimeResources& rr,
    int /* stream_count */
) {
  return all_gc_streams_released(rr);
}

inline void destroy_runtime_resources(RuntimeResources& rr) {
  RA_CUDA_CHECK(cudaDeviceSynchronize());

  for (auto& s : rr.streams) {
    if (s != nullptr) {
      RA_CUDA_CHECK(cudaStreamDestroy(s));
      s = nullptr;
    }
  }

  for (auto& ctx : rr.green_ctxs) {
    if (ctx != nullptr) {
      RA_CUDA_CHECK(cudaExecutionCtxDestroy(ctx));
      ctx = nullptr;
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