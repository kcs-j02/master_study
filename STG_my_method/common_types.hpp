#pragma once

#include <unordered_map>
#include <vector>

#include "../stg_common.hpp"

using stg::KernelKind;
using stg::StgGraph;
using stg::StgTask;
using stg::TaskSpec;

struct NodeInfo {
  int id = -1;
  int indeg = 0;
  std::vector<int> preds;
  std::vector<int> succs;
  int level = -1;
  int stream_id = -1;
  long long proc_time = 0;
};

struct TaskDFG {
  std::unordered_map<int, NodeInfo> nodes;
  std::vector<int> order;
  int num_edges = 0;
};
