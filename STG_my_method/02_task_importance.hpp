#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "00_pipeline_configuration.hpp"
#include "01_stg_analysis.hpp"

/*
 * ================================================================
 * Stage 2
 *
 * proc_timeと後続タスクだけからbottom levelを算出する。
 * ================================================================
 *
 * BL(i)
 *
 *   = proc_time(i)
 *     + max BL(j)
 *
 *       j in succ(i)
 *
 * SM数は重要度に反映しない。
 * ================================================================
 */

struct TaskImportanceResult {
  std::unordered_map<
      int,
      std::vector<int>
  > successors;

  std::unordered_map<
      int,
      double
  > bottom_levels;
};


using TaskTable =
    std::unordered_map<
        int,
        const TaskSpec*
    >;


/*
 * task_id -> TaskSpec
 */
inline TaskTable make_task_table(
    const std::vector<TaskSpec>& tasks
) {
  TaskTable task_by_id;

  task_by_id.reserve(
      tasks.size()
  );


  for (const auto& task : tasks) {
    const auto [it, inserted] =
        task_by_id.emplace(
            task.id,
            &task
        );


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
 * ================================================================
 * タスクの基準処理時間
 * ================================================================
 */
inline std::int64_t
get_task_importance_cost(
    const TaskSpec& task
) {
  /*
   * kernel-aware costを使用する場合
   */
  if (
      std::getenv(
          "STG_KERNEL_AWARE_COST"
      ) != nullptr
  ) {
    const std::int64_t iteration_cost =
        task.kind ==
            KernelKind::HEAVY
            ? 3
            : 1;


    return
        std::max<std::int64_t>(
            0,
            static_cast<std::int64_t>(
                task.work_units
            ) *
            iteration_cost
        );
  }


  /*
   * 通常はSTGのproc_time
   */
  return
      std::max<std::int64_t>(
          0,
          static_cast<std::int64_t>(
              task.proc_time
          )
      );
}


inline std::int64_t
get_task_proc_time(
    const TaskSpec& task
) {
  return
      get_task_importance_cost(
          task
      );
}


namespace task_importance_detail {


using TaskIndex =
    std::unordered_map<
        int,
        std::size_t
    >;


/*
 * task_id -> vector index
 */
inline TaskIndex
make_task_index(
    const std::vector<TaskSpec>& tasks
) {
  TaskIndex task_index;

  task_index.reserve(
      tasks.size()
  );


  for (
      std::size_t index = 0;
      index < tasks.size();
      ++index
  ) {
    const auto [it, inserted] =
        task_index.emplace(
            tasks.at(index).id,
            index
        );


    if (!inserted) {
      throw std::runtime_error(
          "duplicate task id: " +
          std::to_string(
              tasks.at(index).id
          )
      );
    }
  }


  return task_index;
}


/*
 * ================================================================
 * successor表を作る
 * ================================================================
 */
inline std::unordered_map<
    int,
    std::vector<int>
>
make_successor_table(
    const std::vector<TaskSpec>& tasks,
    const TaskIndex& task_index
) {
  std::unordered_map<
      int,
      std::vector<int>
  > successors;


  successors.reserve(
      tasks.size()
  );


  for (const auto& task : tasks) {
    successors.emplace(
        task.id,
        std::vector<int>{}
    );
  }


  for (const auto& task : tasks) {
    for (
        const int pred_id :
        task.preds
    ) {
      /*
       * 自己依存
       */
      if (
          pred_id ==
          task.id
      ) {
        throw std::runtime_error(
            "self dependency found: " +
            std::to_string(
                task.id
            )
        );
      }


      /*
       * 存在しない先行タスク
       */
      if (
          task_index.find(
              pred_id
          ) ==
          task_index.end()
      ) {
        throw std::runtime_error(
            "predecessor task not found: " +
            std::to_string(pred_id) +
            " -> " +
            std::to_string(task.id)
        );
      }


      successors.at(
          pred_id
      ).push_back(
          task.id
      );
    }
  }


  return successors;
}


/*
 * ================================================================
 * bottom levelを再帰的に計算
 * ================================================================
 */
inline double
calculate_bottom_level(
    int task_id,

    const std::unordered_map<
        int,
        std::vector<int>
    >& successors,

    const std::unordered_map<
        int,
        double
    >& proc_times,

    std::unordered_map<
        int,
        int
    >& visit_state,

    std::unordered_map<
        int,
        double
    >& bottom_levels
) {
  const int state =
      visit_state[
          task_id
      ];


  /*
   * 計算済み
   */
  if (
      state == 2
  ) {
    return
        bottom_levels.at(
            task_id
        );
  }


  /*
   * DFS中に再訪したらcycle
   */
  if (
      state == 1
  ) {
    throw std::runtime_error(
        "cycle detected around task: " +
        std::to_string(task_id)
    );
  }


  visit_state[
      task_id
  ] = 1;


  const auto succ_it =
      successors.find(
          task_id
      );


  if (
      succ_it ==
      successors.end()
  ) {
    throw std::runtime_error(
        "successor information not found: " +
        std::to_string(task_id)
    );
  }


  const auto time_it =
      proc_times.find(
          task_id
      );


  if (
      time_it ==
      proc_times.end()
  ) {
    throw std::runtime_error(
        "processing time not found: " +
        std::to_string(task_id)
    );
  }


  double longest_successor_path =
      0.0;


  for (
      const int succ_id :
      succ_it->second
  ) {
    longest_successor_path =
        std::max(
            longest_successor_path,

            calculate_bottom_level(
                succ_id,
                successors,
                proc_times,
                visit_state,
                bottom_levels
            )
        );
  }


  const double bottom_level =
      time_it->second +
      longest_successor_path;


  bottom_levels[
      task_id
  ] = bottom_level;


  visit_state[
      task_id
  ] = 2;


  return bottom_level;
}


/*
 * 全タスクのbottom level
 */
inline std::unordered_map<
    int,
    double
>
calculate_bottom_levels(
    const std::vector<TaskSpec>& tasks,

    const std::unordered_map<
        int,
        std::vector<int>
    >& successors,

    const std::unordered_map<
        int,
        double
    >& proc_times
) {
  std::unordered_map<
      int,
      int
  > visit_state;


  std::unordered_map<
      int,
      double
  > bottom_levels;


  visit_state.reserve(
      tasks.size()
  );


  bottom_levels.reserve(
      tasks.size()
  );


  for (const auto& task : tasks) {
    calculate_bottom_level(
        task.id,
        successors,
        proc_times,
        visit_state,
        bottom_levels
    );
  }


  return bottom_levels;
}


}  // namespace task_importance_detail


/*
 * ================================================================
 * Stage 2本体
 * ================================================================
 */
inline TaskImportanceResult
evaluate_task_importance(
    const std::vector<TaskSpec>& tasks
) {
  if (
      tasks.empty()
  ) {
    throw std::invalid_argument(
        "tasks must not be empty"
    );
  }

  const auto task_index =
      task_importance_detail::
          make_task_index(
              tasks
          );


  TaskImportanceResult result;

  /*
   * successor
   */
  result.successors =
      task_importance_detail::
          make_successor_table(
              tasks,
              task_index
          );


  /*
   * 各タスクのproc_time。
   * bottom level計算中だけ使用し、結果には保持しない。
   */
  std::unordered_map<int, double>
      proc_times;


  proc_times.reserve(
      tasks.size()
  );


  for (const auto& task : tasks) {
    proc_times.emplace(
        task.id,
        static_cast<double>(
            get_task_importance_cost(
                task
            )
        )
    );
  }


  /*
   * bottom level
   */
  result.bottom_levels =
      task_importance_detail::
          calculate_bottom_levels(
              tasks,
              result.successors,
              proc_times
          );


  return result;
}
