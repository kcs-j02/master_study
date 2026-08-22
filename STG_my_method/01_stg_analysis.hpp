#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "00_pipeline_configuration.hpp"
#include "../stg_common.hpp"

using stg::KernelKind;
using stg::StgGraph;
using stg::StgTask;
using stg::TaskSpec;

/*
 * Stage 1: STGを解析する。
 *
 * - 各タスクのproc_timeを取得
 * - 先行・後続関係を構築
 * - level構造から並列性を解析
 */
struct NodeInfo {
  int id = -1;
  int indeg = 0;
  std::vector<int> preds;
  std::vector<int> succs;
  int level = -1;
  long long proc_time = 0;
};

struct TaskDFG {
  std::unordered_map<int, NodeInfo> nodes;
  std::vector<int> order;
  int num_edges = 0;
};

using TaskLevels = std::vector<std::vector<int>>;

struct StgAnalysisResult {
  StgGraph graph;
  std::vector<TaskSpec> tasks;
  TaskDFG dfg;
  TaskLevels levels;
};

namespace stg_analysis_detail {

inline TaskDFG construct_task_dfg(
    const std::vector<TaskSpec>& tasks
) {
  TaskDFG dfg;
  dfg.nodes.reserve(tasks.size());
  dfg.order.reserve(tasks.size());

  for (const auto& task : tasks) {
    NodeInfo node;
    node.id = task.id;
    node.indeg = static_cast<int>(task.preds.size());
    node.preds = task.preds;
    node.proc_time = task.proc_time;

    const auto [it, inserted] = dfg.nodes.emplace(
        task.id,
        std::move(node)
    );

    if (!inserted) {
      throw std::runtime_error(
          "duplicate task id in DFG: " + std::to_string(task.id)
      );
    }

    dfg.order.push_back(task.id);
  }

  for (const auto& task : tasks) {
    for (const int pred_id : task.preds) {
      const auto pred_it = dfg.nodes.find(pred_id);

      if (pred_it == dfg.nodes.end()) {
        throw std::runtime_error(
            "unknown predecessor id in DFG: " +
            std::to_string(pred_id)
        );
      }

      pred_it->second.succs.push_back(task.id);
      ++dfg.num_edges;
    }
  }

  return dfg;
}

inline TaskLevels levelize_task_dfg(TaskDFG& dfg) {
  TaskLevels levels;
  std::unordered_map<int, int> remaining_indeg;
  remaining_indeg.reserve(dfg.nodes.size());

  for (const int task_id : dfg.order) {
    remaining_indeg.emplace(task_id, dfg.nodes.at(task_id).indeg);
  }

  std::size_t assigned_count = 0;

  while (assigned_count < dfg.nodes.size()) {
    std::vector<int> level;

    for (const int task_id : dfg.order) {
      NodeInfo& node = dfg.nodes.at(task_id);

      if (node.level == -1 && remaining_indeg.at(task_id) == 0) {
        node.level = static_cast<int>(levels.size());
        level.push_back(task_id);
      }
    }

    if (level.empty()) {
      throw std::runtime_error(
          "cycle detected during levelization"
      );
    }

    for (const int task_id : level) {
      for (const int succ_id : dfg.nodes.at(task_id).succs) {
        auto it = remaining_indeg.find(succ_id);

        if (it == remaining_indeg.end()) {
          throw std::runtime_error(
              "unknown successor id while levelizing: " +
              std::to_string(succ_id)
          );
        }

        --it->second;

        if (it->second < 0) {
          throw std::runtime_error(
              "invalid indegree while levelizing task: " +
              std::to_string(succ_id)
          );
        }
      }
    }

    assigned_count += level.size();
    levels.push_back(std::move(level));
  }

  return levels;
}

}  // namespace stg_analysis_detail

inline int get_max_level_width(const TaskLevels& levels) {
  int max_width = 0;

  for (const auto& level : levels) {
    max_width = std::max(
        max_width,
        static_cast<int>(level.size())
    );
  }

  if (max_width <= 0) {
    throw std::runtime_error("task levels are empty");
  }

  return max_width;
}

inline int decide_stream_count(
    const TaskLevels& levels,
    int max_stream_count
) {
  if (max_stream_count <= 0 ||
      max_stream_count > kMaximumConfiguredStreamCount) {
    throw std::invalid_argument(
        "max_stream_count must be between 1 and 5"
    );
  }

  return std::min(
      get_max_level_width(levels),
      max_stream_count
  );
}

inline int decide_stream_count(
    const TaskLevels& levels,
    const PipelineOptions& options
) {
  return decide_stream_count(levels, options.max_stream_count);
}

inline StgAnalysisResult analyze_stg(const std::string& path) {
  StgAnalysisResult result;

  result.graph = stg::load_stg_without_comm(path);

  result.tasks = stg::make_task_specs_from_stg_common<TaskSpec>(
      result.graph,
      [](TaskSpec& spec, const StgTask& task, int threshold) {
        spec.parallel_sm_limit = kTaskParallelSmLimit;

        if (task.proc_time > threshold) {
          spec.kind = KernelKind::HEAVY;
          spec.work_units = stg::safe_work_units(
              task.proc_time,
              200,
              1000
          );
        }
        else {
          spec.kind = KernelKind::LIGHT;
          spec.work_units = stg::safe_work_units(
              task.proc_time,
              80,
              200
          );
        }
      }
  );

  result.dfg = stg_analysis_detail::construct_task_dfg(result.tasks);
  result.levels = stg_analysis_detail::levelize_task_dfg(result.dfg);

  return result;
}
