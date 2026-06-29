#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "common_types.hpp"
#include "task_levelization.hpp"
#include <iostream>

inline int decide_stream_count(const TaskLevels& levels, int max_stream_count) {
  int max_width = 0;

  for (const auto& level : levels) {
    max_width = std::max(max_width, static_cast<int>(level.size()));
    std::cout << "max_width: " << max_width << std::endl;
  }

  int stream_count = std::min(max_width, max_stream_count);

  if (stream_count <= 0) {
    throw std::runtime_error("stream_count must be positive");
  }

  return stream_count;
}

inline void assign_stream_to_node(
    const TaskLevels& levels,
    TaskDFG& dfg,
    int stream_count
) {
  if (stream_count <= 0) {
    throw std::runtime_error("stream_count must be positive");
  }

  for (const auto& level : levels) {
    for (size_t i = 0; i < level.size(); ++i) {
      int task_id = level[i];

      NodeInfo& node = dfg.nodes.at(task_id);

      node.stream_id = static_cast<int>(i % stream_count);
    }
  }
}

inline int task_assignment(
    const TaskLevels& levels,
    TaskDFG& dfg,
    int max_stream_count
) {
  int stream_count = decide_stream_count(levels, max_stream_count);

  assign_stream_to_node(levels, dfg, stream_count);

  return stream_count;
}