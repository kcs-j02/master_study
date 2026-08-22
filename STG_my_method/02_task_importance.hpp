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
 * SM配分を考慮した予測bottom levelを算出する。
 * ================================================================
 *
 * 重要:
 *
 *   kSchedulingReferenceSmCount = 114
 *     GPU全体のSM数
 *
 *   kTaskParallelSmLimit = 114
 *     1タスクに許可する最大SM数
 *
 *   kProcTimeReferenceSmCount = 64
 *     proc_timeを取得した基準SM数
 *
 *
 * 現在の予測モデルでは、
 * 64 SMより多く与えても追加の理想速度向上を仮定しない。
 *
 *
 * 例:
 *
 *   8 SM
 *     -> base * 64 / 8
 *
 *   16 SM
 *     -> base * 64 / 16
 *
 *   32 SM
 *     -> base * 64 / 32
 *
 *   64 SM
 *     -> base
 *
 *   82 SM
 *     -> base
 *
 *   114 SM
 *     -> base
 *
 *
 * 各タスクについて各Stream上の予測実行時間を求め、
 * その平均を代表予測時間とする。
 *
 *
 * BL_hat(i)
 *
 *   = p_hat(i)
 *     + max BL_hat(j)
 *
 *       j in succ(i)
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
  > predicted_proc_times;

  std::unordered_map<
      int,
      double
  > bottom_levels;

  std::vector<int>
      stream_sm_counts;
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


/*
 * ================================================================
 * SM数を考慮した予測実行時間
 * ================================================================
 *
 * reference_sm_count:
 *
 *   GPU側で利用可能なSM数の上限。
 *   通常114。
 *
 *
 * proc_timeの基準はreference_sm_countではなく
 *
 *   kProcTimeReferenceSmCount = 64
 *
 * とする。
 *
 *
 * したがって、
 *
 *   GPU全体SM数       = 114
 *   タスク最大SM数     = 114
 *   proc_time基準SM数  = 64
 *
 * を別々に扱う。
 */
inline double
estimate_task_proc_time_on_stream(
    const TaskSpec& task,
    int stream_sm_count,
    int reference_sm_count =
        kSchedulingReferenceSmCount
) {
  if (
      stream_sm_count <= 0
  ) {
    throw std::invalid_argument(
        "stream_sm_count must be positive"
    );
  }


  if (
      reference_sm_count <= 0
  ) {
    throw std::invalid_argument(
        "reference_sm_count must be positive"
    );
  }


  if (
      kProcTimeReferenceSmCount <= 0
  ) {
    throw std::logic_error(
        "kProcTimeReferenceSmCount must be positive"
    );
  }


  const double base_proc_time =
      static_cast<double>(
          get_task_proc_time(
              task
          )
      );


  if (
      base_proc_time <= 0.0
  ) {
    return 0.0;
  }


  /*
   * タスク自身が利用可能な最大SM数。
   */
  const int task_sm_limit =
      std::max(
          1,
          task.parallel_sm_limit
      );


  /*
   * ------------------------------------------------------------
   * proc_time基準側の有効SM数
   * ------------------------------------------------------------
   *
   * 通常:
   *
   *   min(64, 114)
   *   = 64
   */
  const int base_effective_sm =
      std::min(
          kProcTimeReferenceSmCount,
          task_sm_limit
      );


  /*
   * ------------------------------------------------------------
   * Stream側の予測有効SM数
   * ------------------------------------------------------------
   *
   * 以下の最小値を取る。
   *
   *   stream_sm_count
   *   GPU全体SM数
   *   タスクSM上限
   *   proc_time基準SM数
   *
   *
   * 最後に64を含めることで、
   *
   *   64
   *   82
   *   90
   *   114
   *
   * は予測上すべて64 SM相当として扱う。
   */
  const int stream_effective_sm =
      std::max(
          1,
          std::min({
              stream_sm_count,
              reference_sm_count,
              task_sm_limit,
              kProcTimeReferenceSmCount
          })
      );


  /*
   * ------------------------------------------------------------
   * 予測実行時間
   * ------------------------------------------------------------
   */
  return
      base_proc_time *
      static_cast<double>(
          base_effective_sm
      ) /
      static_cast<double>(
          stream_effective_sm
      );
}


/*
 * ================================================================
 * Stage 2用の代表予測実行時間
 * ================================================================
 *
 * SM配分候補に含まれる全Stream上での
 * 予測実行時間を平均する。
 */
inline double
estimate_representative_task_proc_time(
    const TaskSpec& task,

    const std::vector<int>&
        stream_sm_counts,

    int reference_sm_count =
        kSchedulingReferenceSmCount
) {
  validate_stream_sm_counts(
      stream_sm_counts
  );


  if (
      reference_sm_count <= 0
  ) {
    throw std::invalid_argument(
        "reference_sm_count must be positive"
    );
  }


  double total =
      0.0;


  for (
      const int sm_count :
      stream_sm_counts
  ) {
    total +=
        estimate_task_proc_time_on_stream(
            task,
            sm_count,
            reference_sm_count
        );
  }


  return
      total /
      static_cast<double>(
          stream_sm_counts.size()
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
 * 予測bottom levelを再帰的に計算
 * ================================================================
 */
inline double
calculate_predicted_bottom_level(
    int task_id,

    const std::unordered_map<
        int,
        std::vector<int>
    >& successors,

    const std::unordered_map<
        int,
        double
    >& predicted_proc_times,

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
      predicted_proc_times.find(
          task_id
      );


  if (
      time_it ==
      predicted_proc_times.end()
  ) {
    throw std::runtime_error(
        "predicted processing time not found: " +
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

            calculate_predicted_bottom_level(
                succ_id,
                successors,
                predicted_proc_times,
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
 * 全タスクの予測bottom level
 */
inline std::unordered_map<
    int,
    double
>
calculate_predicted_bottom_levels(
    const std::vector<TaskSpec>& tasks,

    const std::unordered_map<
        int,
        std::vector<int>
    >& successors,

    const std::unordered_map<
        int,
        double
    >& predicted_proc_times
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
    calculate_predicted_bottom_level(
        task.id,
        successors,
        predicted_proc_times,
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
 *
 * SM配分候補ごとに呼び出す。
 */
inline TaskImportanceResult
evaluate_task_importance(
    const std::vector<TaskSpec>& tasks,

    const std::vector<int>&
        stream_sm_counts,

    int reference_sm_count =
        kSchedulingReferenceSmCount
) {
  if (
      tasks.empty()
  ) {
    throw std::invalid_argument(
        "tasks must not be empty"
    );
  }


  validate_stream_sm_counts(
      stream_sm_counts
  );


  if (
      reference_sm_count <= 0
  ) {
    throw std::invalid_argument(
        "reference_sm_count must be positive"
    );
  }


  const auto task_index =
      task_importance_detail::
          make_task_index(
              tasks
          );


  TaskImportanceResult result;


  /*
   * Stage 3で同じSM配分を使っているか確認するため保持。
   */
  result.stream_sm_counts =
      stream_sm_counts;


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
   * 各タスクの代表予測実行時間
   */
  result.predicted_proc_times.reserve(
      tasks.size()
  );


  for (const auto& task : tasks) {
    result.predicted_proc_times.emplace(
        task.id,

        estimate_representative_task_proc_time(
            task,
            stream_sm_counts,
            reference_sm_count
        )
    );
  }


  /*
   * 予測bottom level
   */
  result.bottom_levels =
      task_importance_detail::
          calculate_predicted_bottom_levels(
              tasks,
              result.successors,
              result.predicted_proc_times
          );


  return result;
}