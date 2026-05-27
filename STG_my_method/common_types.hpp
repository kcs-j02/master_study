#pragma once

#include <unordered_map>
#include <vector>

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
  int work_units = 1000;
  std::vector<int> preds;
};

struct NodeInfo {
  int id = -1;
  int indeg = 0;
  std::vector<int> preds;
  std::vector<int> succs;
  int level = -1;
  int stream_id = -1;
};

struct TaskDFG {
  std::unordered_map<int, NodeInfo> nodes;
  std::vector<int> order;
  int num_edges = 0;
};