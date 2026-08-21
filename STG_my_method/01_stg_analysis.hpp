#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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
  long long proc_time = 0;
};

struct TaskDFG {
  std::unordered_map<int, NodeInfo> nodes;
  std::vector<int> order;
  int num_edges = 0;
};

using TaskLevels = std::vector<std::vector<int>>;

/*
 * Stage 1: STGの解析
 *
 * タスクkernelが同時に利用できるSM数と、そのSM数に合わせた
 * 要素数をここで一元管理する。kTaskElementCountはStage 5の
 * GPU実行時にも使用する。
 */
inline constexpr int kTaskParallelSmLimit = 64;
inline constexpr int kTaskElementCount =
    kTaskParallelSmLimit * 256;

/*
 * STGの読込み、TaskSpecへの変換、DFG構築、レベル化をまとめた結果。
 */
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

  for (const auto& task : tasks) {
    NodeInfo node;
    node.id = task.id;
    node.preds = task.preds;
    node.indeg = static_cast<int>(task.preds.size());
    node.proc_time = task.proc_time;

    if (!dfg.nodes.emplace(task.id, std::move(node)).second) {
      throw std::runtime_error(
          "duplicate task id in DFG: " + std::to_string(task.id)
      );
    }

    dfg.order.push_back(task.id);
  }

  for (const auto& task : tasks) {
    for (const int predecessor_id : task.preds) {
      const auto predecessor_it = dfg.nodes.find(predecessor_id);
      if (predecessor_it == dfg.nodes.end()) {
        throw std::runtime_error(
            "unknown predecessor id in DFG: " +
            std::to_string(predecessor_id)
        );
      }

      predecessor_it->second.succs.push_back(task.id);
      ++dfg.num_edges;
    }
  }

  return dfg;
}

inline TaskLevels levelize_task_dfg(TaskDFG& dfg) {
  TaskLevels levels;
  std::size_t assigned_count = 0;

  while (assigned_count < dfg.nodes.size()) {
    std::vector<int> level;

    for (const int task_id : dfg.order) {
      NodeInfo& node = dfg.nodes.at(task_id);
      if (node.level == -1 && node.indeg == 0) {
        node.level = static_cast<int>(levels.size());
        level.push_back(task_id);
      }
    }

    if (level.empty()) {
      throw std::runtime_error("cycle detected during levelization");
    }

    for (const int task_id : level) {
      for (const int successor_id : dfg.nodes.at(task_id).succs) {
        NodeInfo& successor = dfg.nodes.at(successor_id);
        --successor.indeg;
        if (successor.indeg < 0) {
          throw std::runtime_error(
              "invalid indegree while levelizing task " +
              std::to_string(successor_id)
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

/*
 * STGファイルを読み込み、後段の各Stageが使用する解析結果を作る。
 */
inline StgAnalysisResult analyze_stg(
    const std::string& path
) {
  StgAnalysisResult result;

  result.graph =
      stg::load_stg_without_comm(path);

  result.tasks =
      stg::make_task_specs_from_stg_common<TaskSpec>(
          result.graph,
          [](TaskSpec& spec,
             const StgTask& task,
             int threshold) {
            spec.parallel_sm_limit =
                kTaskParallelSmLimit;

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

  result.dfg =
      stg_analysis_detail::construct_task_dfg(result.tasks);

  result.levels =
      stg_analysis_detail::levelize_task_dfg(result.dfg);

  return result;
}
