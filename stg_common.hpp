#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    const cudaError_t err__ = (expr);                                         \
    if (err__ != cudaSuccess) {                                               \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__)               \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";         \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)
#endif

namespace stg {

enum class KernelKind {
  LIGHT,
  HEAVY
};

struct StgTask {
  int id = -1;
  int proc_time = 0;
  std::vector<int> preds;
};

struct StgGraph {
  int num_tasks = 0;
  std::vector<StgTask> tasks;
};

struct TaskSpec {
  int id = -1;
  int proc_time = 0;
  KernelKind kind = KernelKind::LIGHT;
  int work_units = 1;
  int parallel_sm_limit = 64;
  std::vector<int> preds;
};

inline std::string trim(const std::string& s) {
  const std::size_t first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }

  const std::size_t last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

inline std::string remove_comment(const std::string& line) {
  std::size_t comment_pos = std::string::npos;
  const std::size_t hash_pos = line.find('#');
  const std::size_t slash_pos = line.find("//");

  if (hash_pos != std::string::npos) {
    comment_pos = hash_pos;
  }

  if (slash_pos != std::string::npos) {
    if (comment_pos == std::string::npos) {
      comment_pos = slash_pos;
    } else {
      comment_pos = std::min(comment_pos, slash_pos);
    }
  }

  return trim(line.substr(0, comment_pos));
}

inline StgGraph load_stg_without_comm(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("failed to open STG file: " + path);
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(ifs, line)) {
    line = remove_comment(line);
    if (!line.empty()) {
      lines.push_back(line);
    }
  }

  if (lines.empty()) {
    throw std::runtime_error("empty STG file: " + path);
  }

  StgGraph graph;

  {
    std::istringstream iss(lines.front());
    if (!(iss >> graph.num_tasks) || graph.num_tasks < 0) {
      throw std::runtime_error("failed to parse number of tasks");
    }
  }

  if (static_cast<int>(lines.size()) < graph.num_tasks + 1) {
    throw std::runtime_error("STG file has fewer task lines than expected");
  }

  graph.tasks.reserve(graph.num_tasks);

  for (int i = 0; i < graph.num_tasks; ++i) {
    std::istringstream iss(lines[i + 1]);
    StgTask task;
    int pred_count = 0;

    if (!(iss >> task.id >> task.proc_time >> pred_count)) {
      throw std::runtime_error("failed to parse task header");
    }

    if (task.id < 0) {
      throw std::runtime_error("negative task id");
    }

    if (task.proc_time < 0) {
      throw std::runtime_error("negative processing time at task " +
                               std::to_string(task.id));
    }

    if (pred_count < 0) {
      throw std::runtime_error("negative predecessor count at task " +
                               std::to_string(task.id));
    }

    task.preds.resize(pred_count);
    for (int k = 0; k < pred_count; ++k) {
      if (!(iss >> task.preds[k])) {
        throw std::runtime_error("failed to parse predecessor list at task " +
                                 std::to_string(task.id));
      }
    }

    graph.tasks.push_back(std::move(task));
  }

  std::vector<int> ids;
  ids.reserve(graph.tasks.size());
  for (const auto& task : graph.tasks) {
    ids.push_back(task.id);
  }

  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
    throw std::runtime_error("duplicate task ids found in STG");
  }

  return graph;
}

inline int safe_work_units(int proc_time, int scale, int minimum) {
  const long long value = static_cast<long long>(std::max(1, proc_time)) * scale;
  return static_cast<int>(std::min<long long>(
      std::max<long long>(minimum, value), std::numeric_limits<int>::max()));
}

template <typename TaskSpecT>
inline std::vector<TaskSpecT> make_task_specs_from_stg_common(
    const StgGraph& graph,
    const std::function<void(TaskSpecT&, const StgTask&, int threshold)>& configure) {
  std::vector<int> positive_times;
  positive_times.reserve(graph.tasks.size());
  for (const auto& task : graph.tasks) {
    if (task.proc_time > 0) {
      positive_times.push_back(task.proc_time);
    }
  }

  int threshold = 1;
  if (!positive_times.empty()) {
    std::sort(positive_times.begin(), positive_times.end());
    threshold = positive_times[positive_times.size() / 2];
  }

  std::vector<TaskSpecT> specs;
  specs.reserve(graph.tasks.size());

  for (const auto& task : graph.tasks) {
    TaskSpecT spec;
    spec.id = task.id;
    spec.proc_time = task.proc_time;
    spec.preds = task.preds;
    configure(spec, task, threshold);
    specs.push_back(std::move(spec));
  }

  return specs;
}

inline __global__ void light_kernel(float* data, int n, int iters) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  float x = data[idx];
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    x = x * 1.000001f + 0.00001f;
  }

  data[idx] = x;
}

inline __global__ void heavy_kernel(float* data, int n, int iters) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  float x = data[idx];
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    x = x * 1.000001f + 0.00001f;
    x = x * 0.999999f + 0.00002f;
    x = x * 1.0000003f - 0.00001f;
  }

  data[idx] = x;
}

template <typename TaskSpecT>
inline void launch_task_kernel(const TaskSpecT& task,
                               float* dmem,
                               int n,
                               cudaStream_t stream) {
  constexpr int THREADS = 256;
  const int blocks = (n + THREADS - 1) / THREADS;

  if (task.kind == KernelKind::HEAVY) {
    heavy_kernel<<<blocks, THREADS, 0, stream>>>(dmem, n, task.work_units);
  } else {
    light_kernel<<<blocks, THREADS, 0, stream>>>(dmem, n, task.work_units);
  }

  CUDA_CHECK(cudaGetLastError());
}

template <typename TaskSpecT>
inline std::vector<std::size_t> make_topological_order(
    const std::vector<TaskSpecT>& tasks) {
  std::unordered_map<int, std::size_t> index_by_id;
  index_by_id.reserve(tasks.size());

  for (std::size_t i = 0; i < tasks.size(); ++i) {
    if (!index_by_id.emplace(tasks[i].id, i).second) {
      throw std::runtime_error("duplicate task id: " + std::to_string(tasks[i].id));
    }
  }

  std::vector<int> indegree(tasks.size(), 0);
  std::vector<std::vector<std::size_t>> successors(tasks.size());

  for (std::size_t i = 0; i < tasks.size(); ++i) {
    indegree[i] = static_cast<int>(tasks[i].preds.size());
    for (const int pred_id : tasks[i].preds) {
      const auto pred_it = index_by_id.find(pred_id);
      if (pred_it == index_by_id.end()) {
        throw std::runtime_error("unknown predecessor id " + std::to_string(pred_id) +
                                 " referenced by task " + std::to_string(tasks[i].id));
      }
      successors[pred_it->second].push_back(i);
    }
  }

  using ReadyItem = std::pair<int, std::size_t>;
  std::priority_queue<ReadyItem, std::vector<ReadyItem>, std::greater<ReadyItem>> ready;

  for (std::size_t i = 0; i < tasks.size(); ++i) {
    if (indegree[i] == 0) {
      ready.emplace(tasks[i].id, i);
    }
  }

  std::vector<std::size_t> order;
  order.reserve(tasks.size());

  while (!ready.empty()) {
    const std::size_t current = ready.top().second;
    ready.pop();
    order.push_back(current);
    for (const std::size_t next : successors[current]) {
      --indegree[next];
      if (indegree[next] == 0) {
        ready.emplace(tasks[next].id, next);
      }
    }
  }

  if (order.size() != tasks.size()) {
    throw std::runtime_error("cycle detected in STG graph");
  }

  return order;
}

template <typename TaskSpecT>
inline void print_stg_summary(const std::vector<TaskSpecT>& tasks) {
  int light_count = 0;
  int heavy_count = 0;
  long long total_proc_time = 0;

  for (const auto& task : tasks) {
    total_proc_time += task.proc_time;
    if (task.kind == KernelKind::HEAVY) {
      ++heavy_count;
    } else {
      ++light_count;
    }
  }

  std::cout << "===== STG summary =====\n"
            << "num_tasks     : " << tasks.size() << "\n"
            << "light_tasks   : " << light_count << "\n"
            << "heavy_tasks   : " << heavy_count << "\n"
            << "sum_proc_time : " << total_proc_time << "\n";
}
}  // namespace stg
