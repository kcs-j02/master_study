#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "01_stg_analysis.hpp"

/*
 * Stage 2: タスク重要度の判定
 *
 * successorsは各タスクの後続タスクID、bottom_levelsは
 * タスク自身から終端までの最長処理時間を保持する。
 */
struct TaskImportanceResult {
  std::unordered_map<int, std::vector<int>> successors;
  std::unordered_map<int, double> bottom_levels;
};

using TaskTable =
    std::unordered_map<int, const TaskSpec*>;

/*
 * task_idからTaskSpecを取得する表を作る。
 *
 * 表内のポインタはtasksの要素を参照するため、tasksの寿命中だけ
 * 有効である。要素の追加など、vectorを再配置する操作を行った後は
 * この表を作り直すこと。
 */
inline TaskTable make_task_table(
    const std::vector<TaskSpec>& tasks
) {
  TaskTable task_by_id;
  task_by_id.reserve(tasks.size());

  for (const auto& task : tasks) {
    const auto [it, inserted] =
        task_by_id.emplace(task.id, &task);

    if (!inserted) {
      throw std::runtime_error(
          "duplicate task id: " +
          std::to_string(task.id)
      );
    }
  }

  return task_by_id;
}

/*
 * proc_timeが0以下のダミータスクはコスト0として扱う。
 * 環境変数STG_KERNEL_AWARE_COSTが設定されている場合は、
 * 実際にkernelへ投入する演算量に近い重みを使用する。
 */
inline std::int64_t get_task_importance_cost(
    const TaskSpec& task
) {
  if (std::getenv("STG_KERNEL_AWARE_COST") != nullptr) {
    const std::int64_t iteration_cost =
        task.kind == KernelKind::HEAVY ? 3 : 1;

    return std::max<std::int64_t>(
        0,
        static_cast<std::int64_t>(task.work_units) *
            iteration_cost
    );
  }

  return std::max<std::int64_t>(
      0,
      static_cast<std::int64_t>(task.proc_time)
  );
}

/*
 * Stage 3以降から利用する処理時間コストAPI。
 */
inline std::int64_t get_task_proc_time(
    const TaskSpec& task
) {
  return get_task_importance_cost(task);
}

namespace task_importance_detail {

using TaskIndex = std::unordered_map<int, std::size_t>;

/*
 * ポインタではなくvector内の添字を保持する。
 * この表は評価中だけ使用し、結果には残さない。
 */
inline TaskIndex make_task_index(
    const std::vector<TaskSpec>& tasks
) {
  TaskIndex task_index;
  task_index.reserve(tasks.size());

  for (std::size_t index = 0;
       index < tasks.size();
       ++index) {
    const auto [it, inserted] =
        task_index.emplace(tasks[index].id, index);

    if (!inserted) {
      throw std::runtime_error(
          "duplicate task id: " +
          std::to_string(tasks[index].id)
      );
    }
  }

  return task_index;
}

inline std::unordered_map<int, std::vector<int>>
make_successor_table(
    const std::vector<TaskSpec>& tasks,
    const TaskIndex& task_index
) {
  std::unordered_map<int, std::vector<int>> successors;
  successors.reserve(tasks.size());

  for (const auto& task : tasks) {
    successors.emplace(task.id, std::vector<int>{});
  }

  for (const auto& task : tasks) {
    for (const int pred_id : task.preds) {
      if (pred_id == task.id) {
        throw std::runtime_error(
            "self dependency found: " +
            std::to_string(task.id)
        );
      }

      if (task_index.find(pred_id) == task_index.end()) {
        throw std::runtime_error(
            "predecessor task not found: " +
            std::to_string(pred_id) +
            " -> " +
            std::to_string(task.id)
        );
      }

      successors.at(pred_id).push_back(task.id);
    }
  }

  return successors;
}

/*
 * bottom_level(task)
 *   = cost(task) + max(bottom_level(successor))
 */
inline double calculate_bottom_level(
    int task_id,
    const std::vector<TaskSpec>& tasks,
    const TaskIndex& task_index,
    const std::unordered_map<int, std::vector<int>>& successors,
    std::unordered_map<int, int>& visit_state,
    std::unordered_map<int, double>& bottom_levels
) {
  const int state = visit_state[task_id];

  if (state == 2) {
    return bottom_levels.at(task_id);
  }

  if (state == 1) {
    throw std::runtime_error(
        "cycle detected around task: " +
        std::to_string(task_id)
    );
  }

  visit_state[task_id] = 1;

  const auto task_it = task_index.find(task_id);

  if (task_it == task_index.end()) {
    throw std::runtime_error(
        "task specification not found: " +
        std::to_string(task_id)
    );
  }

  const auto successor_it = successors.find(task_id);

  if (successor_it == successors.end()) {
    throw std::runtime_error(
        "successor information not found: " +
        std::to_string(task_id)
    );
  }

  double longest_successor_path = 0.0;

  for (const int successor_id : successor_it->second) {
    longest_successor_path = std::max(
        longest_successor_path,
        calculate_bottom_level(
            successor_id,
            tasks,
            task_index,
            successors,
            visit_state,
            bottom_levels
        )
    );
  }

  const TaskSpec& task =
      tasks.at(task_it->second);

  const double bottom_level =
      static_cast<double>(
          get_task_importance_cost(task)
      ) + longest_successor_path;

  bottom_levels[task_id] = bottom_level;
  visit_state[task_id] = 2;

  return bottom_level;
}

inline std::unordered_map<int, double>
calculate_bottom_levels(
    const std::vector<TaskSpec>& tasks,
    const TaskIndex& task_index,
    const std::unordered_map<int, std::vector<int>>& successors
) {
  std::unordered_map<int, int> visit_state;
  std::unordered_map<int, double> bottom_levels;

  visit_state.reserve(tasks.size());
  bottom_levels.reserve(tasks.size());

  for (const auto& task : tasks) {
    calculate_bottom_level(
        task.id,
        tasks,
        task_index,
        successors,
        visit_state,
        bottom_levels
    );
  }

  return bottom_levels;
}

}  // namespace task_importance_detail

/*
 * 後続関係を検証し、全タスクのbottom levelを計算する。
 */
inline TaskImportanceResult evaluate_task_importance(
    const std::vector<TaskSpec>& tasks
) {
  const auto task_index =
      task_importance_detail::make_task_index(tasks);

  TaskImportanceResult result;
  result.successors =
      task_importance_detail::make_successor_table(
          tasks,
          task_index
      );

  result.bottom_levels =
      task_importance_detail::calculate_bottom_levels(
          tasks,
          task_index,
          result.successors
      );

  return result;
}
