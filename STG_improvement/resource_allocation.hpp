#pragma once

#include <cuda_runtime.h>

#include <algorithm>
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

inline void check_resource(int available_sm, int reserve_sm, int unit, int min_group_sm, RuntimeResources rr,  int stream_count){
  std::cout << "===== Green Context Resource Allocation =====\n";
  std::cout << "available SM = " << available_sm << "\n";
  std::cout << "reserve SM   = " << reserve_sm << "\n";
  std::cout << "unit SM      = " << unit << "\n";
  std::cout << "min group SM = " << min_group_sm << "\n";

  for (int i = 0; i < stream_count; ++i) {
    std::cout << "stream[" << i << "] priority = "
              << rr.priority_each_stream[i]
              << ", SM = "
              << rr.sm_count_each_stream[i]
              << "\n";
  }
}

// 実行
inline RuntimeResources resource_allocation(
    const TaskDFG& dfg,
    const std::vector<TaskSpec>& tasks,
    int stream_count,
    int gpu_device_index = 0,
    int reserve_sm = 10
) {
  if (stream_count <= 0) {
    throw std::runtime_error("stream_count must be positive");
  }

  RA_CUDA_CHECK(cudaSetDevice(gpu_device_index));

  RuntimeResources rr;
  rr.streams.resize(stream_count, nullptr);
  rr.green_ctxs.resize(stream_count, nullptr);
  rr.resource_descs.resize(stream_count, nullptr);
  rr.sm_count_each_stream.resize(stream_count, 0);
  rr.priority_each_stream.resize(stream_count, 0);

  // task_id -> order_count
  // ここでは work_units をタスクのオーダー数として使う
  std::unordered_map<int, long long> order_count_by_task;

  for (const auto& t : tasks) {
    long long order_count = std::max(1, t.work_units);

    // heavy_kernel は light_kernel より1ループあたり演算が多い
    if (t.kind == KernelKind::HEAVY) {
      order_count *= 3;
    }

    order_count_by_task[t.id] = order_count;
  }

  // streamごとのpriorityを作る
  for (const auto& [task_id, ni] : dfg.nodes) {
    if (ni.stream_id >= 0 && ni.stream_id < stream_count) {
      auto it = order_count_by_task.find(task_id);

      if (it == order_count_by_task.end()) {
        throw std::runtime_error(
            "task id not found in tasks: " + std::to_string(task_id)
        );
      }

      rr.priority_each_stream[ni.stream_id] += it->second;
    }
  }

  long long total_priority = 0;
  for (long long p : rr.priority_each_stream) {
    total_priority += p;
  }

  if (total_priority == 0) {
    throw std::runtime_error("total priority is zero");
  }

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

  if (available_sm - reserve_sm < stream_count * min_group_sm) {
    reserve_sm = 0;
  }

  int allocatable_sm = available_sm - reserve_sm;

  if (allocatable_sm < stream_count * min_group_sm) {
    throw std::runtime_error(
        "not enough SMs for Green Contexts: available_sm=" +
        std::to_string(available_sm) +
        ", stream_count=" +
        std::to_string(stream_count)
    );
  }

  // 全streamに最低SM数を割り当てる
  for (int i = 0; i < stream_count; ++i) {
    rr.sm_count_each_stream[i] = min_group_sm;
  }

  int remaining_sm = allocatable_sm - stream_count * min_group_sm;
  int remaining_chunks = remaining_sm / unit;

  // priority/order_count が大きいstreamへ残りSMを配分
  for (int c = 0; c < remaining_chunks; ++c) {
    int best_stream = 0;
    long double best_score = -1.0;

    for (int i = 0; i < stream_count; ++i) {
      long double score =
          static_cast<long double>(rr.priority_each_stream[i]) /
          static_cast<long double>(rr.sm_count_each_stream[i]);

      if (score > best_score) {
        best_score = score;
        best_stream = i;
      }
    }

    rr.sm_count_each_stream[best_stream] += unit;
  }

  // check_resource
  check_resource(available_sm, reserve_sm, unit, min_group_sm, rr,  stream_count);

  std::vector<cudaDevResource> split_sm_resources(stream_count);
  std::vector<cudaDevSmResourceGroupParams> group_params(stream_count);

  for (int i = 0; i < stream_count; ++i) {
    group_params[i] = {};
    group_params[i].smCount =
        static_cast<unsigned int>(rr.sm_count_each_stream[i]);
    group_params[i].coscheduledSmCount = 0;
    group_params[i].preferredCoscheduledSmCount = 0;
    group_params[i].flags = 0;
  }

  RA_CUDA_CHECK(cudaDevSmResourceSplit(
      split_sm_resources.data(),
      static_cast<unsigned int>(stream_count),
      &initial_sm_resource,
      nullptr,
      0,
      group_params.data()
  ));

  for (int i = 0; i < stream_count; ++i) {
    RA_CUDA_CHECK(cudaDevResourceGenerateDesc(
        &rr.resource_descs[i],
        &split_sm_resources[i],
        1
    ));
  }

  for (int i = 0; i < stream_count; ++i) {
    RA_CUDA_CHECK(cudaGreenCtxCreate(
        &rr.green_ctxs[i],
        rr.resource_descs[i],
        gpu_device_index,
        0
    ));
  }

  for (int i = 0; i < stream_count; ++i) {
    RA_CUDA_CHECK(cudaExecutionCtxStreamCreate(
        &rr.streams[i],
        rr.green_ctxs[i],
        cudaStreamDefault,
        0
    ));
  }

  return rr;
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
}