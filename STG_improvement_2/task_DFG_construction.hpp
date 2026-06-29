#pragma once

#include <stdexcept>
#include <string>

#include "common_types.hpp"

inline TaskDFG task_DFG_construction(const std::vector<TaskSpec>& tasks) {
  TaskDFG dfg;

  for (const auto& t : tasks) {
    if (dfg.nodes.find(t.id) != dfg.nodes.end()) {
      throw std::runtime_error(
        "duplicate task id in DFG: " + std::to_string(t.id)
      );
    }

    NodeInfo node;
    node.id = t.id;
    node.preds = t.preds;
    node.indeg = static_cast<int>(t.preds.size());

    node.proc_time = t.proc_time;


    dfg.nodes.emplace(t.id, std::move(node));
    dfg.order.push_back(t.id);
  }

  for (const auto& t : tasks) {
    for (int pred : t.preds) {
      auto pred_it = dfg.nodes.find(pred);

      if (pred_it == dfg.nodes.end()) {
        throw std::runtime_error(
          "unknown predecessor id in DFG: " + std::to_string(pred)
        );
      }

      pred_it->second.succs.push_back(t.id);
      dfg.num_edges++;
    }
  }

  return dfg;
}