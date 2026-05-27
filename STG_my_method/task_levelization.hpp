#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common_types.hpp"

using TaskLevels = std::vector<std::vector<int>>;

inline void add_level(TaskLevels& levels, int count_level) {
  if (static_cast<int>(levels.size()) <= count_level) {
    levels.emplace_back();
  }
}

inline void push_node_to_level(
    int count_level,
    TaskDFG& dfg,
    TaskLevels& levels,
    int& node_count
) {
  for (int id : dfg.order) {
    NodeInfo& node = dfg.nodes.at(id);

    if (node.level != -1) {
      continue;
    }

    if (node.indeg == 0) {
      node.level = count_level;
      levels[count_level].push_back(id);
      node_count++;
    }
  }
}

inline void remove_node(
    int count_level,
    TaskDFG& dfg,
    TaskLevels& levels
) {
  for (int id : levels[count_level]) {
    NodeInfo& node = dfg.nodes.at(id);

    for (int succ : node.succs) {
      NodeInfo& succ_node = dfg.nodes.at(succ);
      succ_node.indeg--;
    }
  }
}

inline void print_levels(const TaskLevels& levels) {
  for (size_t i = 0; i < levels.size(); ++i) {
    std::cout << "level[" << i << "] = {";

    for (size_t j = 0; j < levels[i].size(); ++j) {
      if (j) std::cout << ", ";
      std::cout << levels[i][j];
    }

    std::cout << "}\n";
  }
}

inline TaskLevels task_levelization(TaskDFG& dfg) {
  int count_level = 0;
  int node_count = 0;
  int flag = 1;

  TaskLevels levels;

  while (flag) {
    int before_count = node_count;

    add_level(levels, count_level);
    push_node_to_level(count_level, dfg, levels, node_count);
    remove_node(count_level, dfg, levels);

    if (node_count == static_cast<int>(dfg.nodes.size())) {
      flag = 0;
    }

    if (before_count == node_count) {
      throw std::runtime_error("cycle detected during levelization");
    }

    count_level++;
  }

  // print_levels(levels);

  return levels;
}

#include <iostream>

inline void check_nodes(const TaskDFG& dfg) {
  std::cout << "===== Node Info =====\n";

  for (int id : dfg.order) {
    const NodeInfo& ni = dfg.nodes.at(id);

    std::cout << "task " << ni.id
              << " 入次数=" << ni.preds.size()
              << " 依存元=[";

    for (int pred : ni.preds) {
      std::cout << pred << " ";
    }

    std::cout << "] level=" << ni.level
              << " stream=" << ni.stream_id
              << "\n";
  }

  std::cout << "=====================\n";
}