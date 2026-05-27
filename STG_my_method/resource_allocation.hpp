#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "common_types.hpp"

struct RuntimeResources {
  std::vector<cudaStream_t> streams;
  std::vector<cudaExecutionContext_t> green_ctxs;
  std::vector<cudaDevResourceDesc_t> resource_descs;
};

inline void check_cuda(cudaError_t err, const std::string& msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(msg + ": " + cudaGetErrorString(err));
  }
}

inline cudaStream_t get_stream_by_id(RuntimeResources& rr, int stream_id) {
  if (stream_id < 0 || stream_id >= static_cast<int>(rr.streams.size())) {
    throw std::runtime_error("invalid stream_id: " + std::to_string(stream_id));
  }

  return rr.streams[stream_id];
}

inline std::vector<int> count_indegree_each_stream(const TaskDFG& dfg,
                                                   int stream_count) {
  std::vector<int> counts(stream_count, 0);

  for (int id : dfg.order) {
    const NodeInfo& node = dfg.nodes.at(id);

    if (node.stream_id < 0 || node.stream_id >= stream_count) {
      throw std::runtime_error(
        "invalid stream_id at task " + std::to_string(id)
      );
    }

    counts[node.stream_id] += static_cast<int>(node.preds.size());
  }

  return counts;
}

inline std::vector<int> decide_sm_counts(const std::vector<int>& indeg_counts,
                                         int sm_count,
                                         int reserve_sm = 10) {
  int stream_count = static_cast<int>(indeg_counts.size());

  if (stream_count <= 0) {
    throw std::runtime_error("stream_count must be positive");
  }

  int usable_sm = sm_count - reserve_sm;
  if (usable_sm < 2 * stream_count) {
    usable_sm = 2 * stream_count;
  }
  if (usable_sm > sm_count) {
    usable_sm = sm_count;
  }

  // 偶数にする
  usable_sm = (usable_sm / 2) * 2;

  std::vector<int> sm_counts(stream_count, 2);

  int remaining_sm = usable_sm - 2 * stream_count;
  if (remaining_sm < 0) {
    throw std::runtime_error("not enough SMs for stream_count");
  }

  int total_indeg = 0;
  for (int v : indeg_counts) {
    total_indeg += v;
  }

  if (total_indeg == 0) {
    for (int i = 0; i + 2 <= remaining_sm; i += 2) {
      sm_counts[(i / 2) % stream_count] += 2;
    }
    return sm_counts;
  }

  int assigned_extra = 0;

  for (int i = 0; i < stream_count; ++i) {
    int extra = (remaining_sm * indeg_counts[i]) / total_indeg;

    // 偶数に丸める
    extra = (extra / 2) * 2;

    sm_counts[i] += extra;
    assigned_extra += extra;
  }

  int rest = remaining_sm - assigned_extra;

  for (int i = 0; i + 2 <= rest; i += 2) {
    sm_counts[(i / 2) % stream_count] += 2;
  }

  return sm_counts;
}

inline RuntimeResources make_runtime_resources(const TaskDFG& dfg,
                                               int stream_count) {
  if (stream_count <= 0) {
    throw std::runtime_error("stream_count must be positive");
  }

  RuntimeResources rr;

  int gpu_device_index = 0;

  check_cuda(cudaSetDevice(gpu_device_index), "cudaSetDevice failed");

  cudaDeviceProp prop {};
  check_cuda(
    cudaGetDeviceProperties(&prop, gpu_device_index),
    "cudaGetDeviceProperties failed"
  );

  int sm_count = prop.multiProcessorCount;

  std::vector<int> indeg_counts =
    count_indegree_each_stream(dfg, stream_count);

  std::vector<int> sm_counts =
    decide_sm_counts(indeg_counts, sm_count, 10);

  cudaDevResource initial_sm_resource {};

  check_cuda(
    cudaDeviceGetDevResource(
      gpu_device_index,
      &initial_sm_resource,
      cudaDevResourceTypeSm
    ),
    "cudaDeviceGetDevResource failed"
  );

  std::vector<cudaDevResource> split_resources(stream_count);
  std::vector<cudaDevSmResourceGroupParams> group_params(stream_count);

  for (int i = 0; i < stream_count; ++i) {
    group_params[i] = {
      .smCount = static_cast<unsigned int>(sm_counts[i]),
      .coscheduledSmCount = 2u,
      .preferredCoscheduledSmCount = 0u,
      .flags = 0u
    };
  }
  std::cout << "sm_count = " << sm_count << "\n";
  for (int i = 0; i < stream_count; ++i) {
    std::cout << "sm_counts[" << i << "] = " << sm_counts[i] << "\n";
  }
  check_cuda(
    cudaDevSmResourceSplit(
      split_resources.data(),
      stream_count,
      &initial_sm_resource,
      nullptr,
      0,
      group_params.data()
    ),
    "cudaDevSmResourceSplit failed"
  );

  rr.resource_descs.resize(stream_count);
  rr.green_ctxs.resize(stream_count);
  rr.streams.resize(stream_count);

  for (int i = 0; i < stream_count; ++i) {
    check_cuda(
      cudaDevResourceGenerateDesc(
        &rr.resource_descs[i],
        &split_resources[i],
        1
      ),
      "cudaDevResourceGenerateDesc failed"
    );
  }

  for (int i = 0; i < stream_count; ++i) {
    check_cuda(
      cudaGreenCtxCreate(
        &rr.green_ctxs[i],
        rr.resource_descs[i],
        gpu_device_index,
        0
      ),
      "cudaGreenCtxCreate failed"
    );
  }

  for (int i = 0; i < stream_count; ++i) {
    check_cuda(
      cudaExecutionCtxStreamCreate(
        &rr.streams[i],
        rr.green_ctxs[i],
        cudaStreamDefault,
        0
      ),
      "cudaExecutionCtxStreamCreate failed"
    );
  }

  return rr;
}

inline void destroy_runtime_resources(RuntimeResources& rr) {
  for (cudaStream_t s : rr.streams) {
    if (s != nullptr) {
      check_cuda(cudaStreamDestroy(s), "cudaStreamDestroy failed");
    }
  }

  for (cudaExecutionContext_t ctx : rr.green_ctxs) {
    check_cuda(cudaExecutionCtxDestroy(ctx), "cudaExecutionCtxDestroy failed");
  }

  rr.streams.clear();
  rr.green_ctxs.clear();
  rr.resource_descs.clear();
}