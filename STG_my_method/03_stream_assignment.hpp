#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "00_pipeline_configuration.hpp"
#include "01_stg_analysis.hpp"
#include "02_task_importance.hpp"

/*
 * ================================================================
 * Stage 3
 *
 * 予測完了時刻が最小のStreamへ配置する。
 * ================================================================
 *
 * ready-listからbottom level最大のタスクを選択する。
 *
 * 各Streamについて
 *
 *   EFT(i,s)
 *
 *     = EST(i,s)
 *       + proc_time(i)
 *
 * を計算する。
 *
 *
 * タスク配置ではSM数を使用しない。配置を確定した後、その配置を
 * 固定したまま割当SM数を反映した予測makespanを再計算する。
 * ================================================================
 */

struct StreamScheduleResult {
  std::vector<int>
      stream_sm_counts;

  std::unordered_map<
      int,
      int
  > task_stream;

  std::unordered_map<
      int,
      double
  > task_start_time;

  std::unordered_map<
      int,
      double
  > task_finish_time;

  double makespan =
      0.0;
};


struct ScheduledInterval {
  double start_time =
      0.0;

  double finish_time =
      0.0;

  int task_id =
      -1;
};


/*
 * ================================================================
 * Stream内の最初の空き時間を求める
 * ================================================================
 */
inline double
find_earliest_insertion_start(
    const std::vector<ScheduledInterval>& intervals,

    double dependency_ready_time,

    double duration
) {
  constexpr double epsilon =
      1.0e-9;


  double candidate_start =
      dependency_ready_time;


  for (
      const auto& interval :
      intervals
  ) {
    /*
     * 現在位置から次のタスク開始までに
     * duration分の空きがある。
     */
    if (
        candidate_start +
            duration <=
        interval.start_time +
            epsilon
    ) {
      return candidate_start;
    }


    /*
     * 既存タスクと重なる場合は、
     * その終了時刻まで進める。
     */
    if (
        candidate_start <
        interval.finish_time
    ) {
      candidate_start =
          interval.finish_time;
    }
  }


  return candidate_start;
}


/*
 * ================================================================
 * Stream上へ区間を追加
 * ================================================================
 */
inline void
insert_scheduled_interval(
    std::vector<ScheduledInterval>& intervals,

    ScheduledInterval interval
) {
  const auto position =
      std::lower_bound(
          intervals.begin(),
          intervals.end(),
          interval,

          [](
              const ScheduledInterval& lhs,
              const ScheduledInterval& rhs
          ) {
            if (
                lhs.start_time !=
                rhs.start_time
            ) {
              return
                  lhs.start_time <
                  rhs.start_time;
            }


            return
                lhs.task_id <
                rhs.task_id;
          }
      );


  intervals.insert(
      position,
      interval
  );
}


namespace stream_assignment_detail {


/*
 * ================================================================
 * Stage 2結果を検証
 * ================================================================
 */
inline void
validate_importance_result(
    const std::vector<TaskSpec>& tasks,

    const TaskImportanceResult& importance,

    const TaskTable& task_by_id
) {
  for (const auto& task : tasks) {

    /*
     * bottom level
     */
    if (
        importance.bottom_levels.find(
            task.id
        ) ==
        importance.bottom_levels.end()
    ) {
      throw std::invalid_argument(
        "bottom level not found for task: " +
        std::to_string(task.id)
      );
    }


    /*
     * successor
     */
    const auto succ_it =
        importance.successors.find(
            task.id
        );


    if (
        succ_it ==
        importance.successors.end()
    ) {
      throw std::invalid_argument(
          "successor information not found for task: " +
          std::to_string(task.id)
      );
    }


    for (
        const int succ_id :
        succ_it->second
    ) {
      if (
          task_by_id.find(
              succ_id
          ) ==
          task_by_id.end()
      ) {
        throw std::invalid_argument(
            "successor task not found: " +
            std::to_string(task.id) +
            " -> " +
            std::to_string(succ_id)
        );
      }
    }
  }
}


/*
 * ================================================================
 * 全先行タスクが完了する予測時刻
 * ================================================================
 */
inline double
get_predecessor_ready_time(
    const TaskSpec& task,

    const std::unordered_map<
        int,
        double
    >& task_finish_time
) {
  double ready_time =
      0.0;


  for (
      const int pred_id :
      task.preds
  ) {
    const auto finish_it =
        task_finish_time.find(
            pred_id
        );


    if (
        finish_it ==
        task_finish_time.end()
    ) {
      throw std::runtime_error(
          "predecessor has not been scheduled: " +
          std::to_string(pred_id) +
          " -> " +
          std::to_string(task.id)
      );
    }


    ready_time =
        std::max(
            ready_time,
            finish_it->second
        );
  }


  return ready_time;
}


}  // namespace stream_assignment_detail


/*
 * ================================================================
 * 固定済みのタスク配置にSM数を反映してmakespanを再計算する
 * ================================================================
 *
 * task_streamと各Stream内のタスク順序は変更しない。
 * STG依存関係とStream内直列実行を制約として、割当SM数で補正した
 * タスク処理時間から最早開始・終了時刻を求める。
 */
inline double estimate_fixed_assignment_makespan(
    const std::vector<TaskSpec>& tasks,
    const std::vector<int>& stream_sm_counts,
    const StreamScheduleResult& placement,
    int reference_sm_count = kSchedulingReferenceSmCount
) {
  const auto task_by_id = make_task_table(tasks);
  const int stream_count =
      static_cast<int>(stream_sm_counts.size());

  std::vector<std::vector<int>> stream_task_ids(
      static_cast<std::size_t>(stream_count)
  );

  for (const auto& task : tasks) {
    const int stream_id = placement.task_stream.at(task.id);
    stream_task_ids.at(static_cast<std::size_t>(stream_id)).push_back(
        task.id
    );
  }

  for (auto& task_ids : stream_task_ids) {
    std::stable_sort(
        task_ids.begin(),
        task_ids.end(),
        [&](int left_id, int right_id) {
          const double left_start =
              placement.task_start_time.at(left_id);
          const double right_start =
              placement.task_start_time.at(right_id);

          if (left_start != right_start) {
            return left_start < right_start;
          }

          const double left_finish =
              placement.task_finish_time.at(left_id);
          const double right_finish =
              placement.task_finish_time.at(right_id);

          if (left_finish != right_finish) {
            return left_finish < right_finish;
          }

          return left_id < right_id;
        }
    );
  }

  std::unordered_map<int, std::vector<int>> predecessors;
  std::unordered_map<int, std::vector<int>> successors;
  std::unordered_map<int, int> remaining_predecessors;

  for (const auto& task : tasks) {
    predecessors.emplace(task.id, task.preds);
    successors.emplace(task.id, std::vector<int>{});
  }

  /* 同一Stream内の実行順序を追加の依存制約として扱う。 */
  for (const auto& task_ids : stream_task_ids) {
    for (std::size_t index = 1; index < task_ids.size(); ++index) {
      const int previous_id = task_ids.at(index - 1);
      const int current_id = task_ids.at(index);
      auto& current_predecessors = predecessors.at(current_id);

      if (std::find(
              current_predecessors.begin(),
              current_predecessors.end(),
              previous_id
          ) == current_predecessors.end()) {
        current_predecessors.push_back(previous_id);
      }
    }
  }

  for (const auto& task : tasks) {
    const auto& task_predecessors = predecessors.at(task.id);
    remaining_predecessors.emplace(
        task.id,
        static_cast<int>(task_predecessors.size())
    );

    for (const int predecessor_id : task_predecessors) {
      successors.at(predecessor_id).push_back(task.id);
    }
  }

  std::vector<int> ready_task_ids;
  for (const auto& task : tasks) {
    if (remaining_predecessors.at(task.id) == 0) {
      ready_task_ids.push_back(task.id);
    }
  }

  std::unordered_map<int, double> finish_times;
  double makespan = 0.0;
  std::size_t evaluated_task_count = 0;

  while (!ready_task_ids.empty()) {
    const int task_id = ready_task_ids.back();
    ready_task_ids.pop_back();
    const TaskSpec& task = *task_by_id.at(task_id);

    double start_time = 0.0;
    for (const int predecessor_id : predecessors.at(task_id)) {
      start_time = std::max(
          start_time,
          finish_times.at(predecessor_id)
      );
    }

    const int stream_id = placement.task_stream.at(task_id);
    const double duration = estimate_task_proc_time_on_stream(
        task,
        stream_sm_counts.at(static_cast<std::size_t>(stream_id)),
        reference_sm_count
    );
    const double finish_time = start_time + duration;

    finish_times.emplace(task_id, finish_time);
    makespan = std::max(makespan, finish_time);
    ++evaluated_task_count;

    for (const int successor_id : successors.at(task_id)) {
      int& remaining = remaining_predecessors.at(successor_id);
      --remaining;
      if (remaining == 0) {
        ready_task_ids.push_back(successor_id);
      }
    }
  }

  if (evaluated_task_count != tasks.size()) {
    throw std::runtime_error(
        "failed to evaluate fixed stream assignment"
    );
  }

  return makespan;
}


/*
 * ================================================================
 * Stage 3本体
 * ================================================================
 *
 * 1.
 *   ready-listから
 *   bottom level最大のタスクを選択
 *
 * 2.
 *   各StreamのESTを計算
 *
 * 3.
 *   元のproc_timeからEFTを計算
 *
 * 4.
 *   EFT = EST + proc_time
 *
 * 5.
 *   EFT最小のStreamへ配置
 *
 *
 * 同値の場合:
 *
 *   EFT
 *     ↓
 *   EST
 *     ↓
 *   Stream ID
 *
 * の順に比較する。
 *
 *
 * ================================================================
 */
inline StreamScheduleResult
place_tasks_on_streams(
    const std::vector<TaskSpec>& tasks,

    const TaskImportanceResult& importance,

    const std::vector<int>&
        stream_sm_counts,

    int reference_sm_count =
        kSchedulingReferenceSmCount
) {
  /*
   * ------------------------------------------------------------
   * 入力確認
   * ------------------------------------------------------------
   */
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


  const auto task_by_id =
      make_task_table(
          tasks
      );


  stream_assignment_detail::
      validate_importance_result(
          tasks,
          importance,
          task_by_id
      );


  const int stream_count =
      static_cast<int>(
          stream_sm_counts.size()
      );


  /*
   * 各Streamの予定区間
   */
  std::vector<
      std::vector<ScheduledInterval>
  > stream_intervals(
      static_cast<std::size_t>(
          stream_count
      )
  );


  /*
   * 未配置の先行タスク数
   */
  std::unordered_map<
      int,
      int
  > remaining_predecessor_count;


  remaining_predecessor_count.reserve(
      tasks.size()
  );


  /*
   * ready-list
   */
  std::vector<int>
      ready_task_ids;


  ready_task_ids.reserve(
      tasks.size()
  );


  /*
   * ============================================================
   * 初期ready-list
   * ============================================================
   */
  for (const auto& task : tasks) {
    const int pred_count =
        static_cast<int>(
            task.preds.size()
        );


    remaining_predecessor_count.emplace(
        task.id,
        pred_count
    );


    if (
        pred_count == 0
    ) {
      ready_task_ids.push_back(
          task.id
      );
    }
  }


  StreamScheduleResult result;


  /* Stage 4で候補とスケジュールの対応を検証する。 */
  result.stream_sm_counts =
      stream_sm_counts;


  result.task_stream.reserve(
      tasks.size()
  );


  result.task_start_time.reserve(
      tasks.size()
  );


  result.task_finish_time.reserve(
      tasks.size()
  );


  constexpr double epsilon =
      1.0e-9;


  std::size_t scheduled_task_count =
      0;


  /*
   * ============================================================
   * ready-list scheduling
   * ============================================================
   */
  while (
      !ready_task_ids.empty()
  ) {
    /*
     * ----------------------------------------------------------
     * ready-listから
     * bottom level最大のタスクを選ぶ。
     *
     * 同値なら
     *
     *   proc_time
     *   ↓
     *   task ID
     *
     * で決める。
     * ----------------------------------------------------------
     */
    std::size_t selected_index =
        0;


    for (
        std::size_t index = 1;
        index < ready_task_ids.size();
        ++index
    ) {
      const int current_id =
          ready_task_ids.at(
              index
          );


      const int selected_id =
          ready_task_ids.at(
              selected_index
          );


      const double current_bl =
          importance.bottom_levels.at(
              current_id
          );


      const double selected_bl =
          importance.bottom_levels.at(
              selected_id
          );


      bool select_current =
          false;


      /*
       * BLが大きい
       */
      if (
          current_bl >
          selected_bl +
              epsilon
      ) {
        select_current =
            true;
      }


      /*
       * BL同値
       */
      else if (
          std::abs(
              current_bl -
              selected_bl
          ) <= epsilon
      ) {
        const std::int64_t current_time =
            get_task_proc_time(
                *task_by_id.at(
                    current_id
                )
            );


        const std::int64_t selected_time =
            get_task_proc_time(
                *task_by_id.at(
                    selected_id
                )
            );


        /*
         * proc_timeが長い方
         */
        if (
            current_time >
            selected_time
        ) {
          select_current =
              true;
        }


        /*
         * それも同じならID順
         */
        else if (
            current_time ==
                selected_time &&

            current_id <
                selected_id
        ) {
          select_current =
              true;
        }
      }


      if (
          select_current
      ) {
        selected_index =
            index;
      }
    }


    /*
     * 選択タスク
     */
    const int task_id =
        ready_task_ids.at(
            selected_index
        );


    ready_task_ids.erase(
        ready_task_ids.begin() +
        static_cast<std::ptrdiff_t>(
            selected_index
        )
    );


    const TaskSpec& task =
        *task_by_id.at(
            task_id
        );


    /*
     * ----------------------------------------------------------
     * 依存関係によるready時刻
     * ----------------------------------------------------------
     */
    const double dependency_ready_time =
        stream_assignment_detail::
            get_predecessor_ready_time(
                task,
                result.task_finish_time
            );


    /*
     * ----------------------------------------------------------
     * 全StreamのEFTを比較
     * ----------------------------------------------------------
     */
    int best_stream =
        -1;


    double best_start_time =
        std::numeric_limits<double>::max();


    double best_finish_time =
        std::numeric_limits<double>::max();


    for (
        int stream_id = 0;
        stream_id < stream_count;
        ++stream_id
    ) {
      /*
       * ========================================================
       * 配置時はSM数を反映しない元のproc_timeを使用
       * ========================================================
       */
      const double placement_proc_time =
          static_cast<double>(get_task_proc_time(task));


      /*
       * EST
       */
      const double start_time =
          find_earliest_insertion_start(
              stream_intervals.at(
                  static_cast<std::size_t>(
                      stream_id
                  )
              ),
              dependency_ready_time,
              placement_proc_time
          );


      /*
       * EFT
       */
      const double finish_time =
          start_time +
          placement_proc_time;


      bool select_current =
          false;


      /*
       * --------------------------------------------------------
       * 第1基準:
       *
       * EFT最小
       * --------------------------------------------------------
       */
      if (
          finish_time <
          best_finish_time -
              epsilon
      ) {
        select_current =
            true;
      }


      /*
       * EFT同値
       */
      else if (
          std::abs(
              finish_time -
              best_finish_time
          ) <= epsilon
      ) {
        /*
         * ------------------------------------------------------
         * 第2基準:
         *
         * EST最小
         * ------------------------------------------------------
         */
        if (
            start_time <
            best_start_time -
                epsilon
        ) {
          select_current =
              true;
        }


        /*
         * ESTも同値
         */
        else if (
            std::abs(
                start_time -
                best_start_time
            ) <= epsilon
        ) {
          /*
           * ----------------------------------------------------
           * 第3基準:
           *
           * Stream ID
           * ----------------------------------------------------
           */
          if (
              best_stream < 0 ||
              stream_id < best_stream
          ) {
            select_current =
                true;
          }
        }
      }


      if (
          select_current
      ) {
        best_stream =
            stream_id;


        best_start_time =
            start_time;


        best_finish_time =
            finish_time;
      }
    }


    /*
     * Stream未選択
     */
    if (
        best_stream < 0
    ) {
      throw std::runtime_error(
          "failed to select stream for task: " +
          std::to_string(task_id)
      );
    }


    /*
     * ==========================================================
     * タスクを選択Streamへ配置
     * ==========================================================
     */
    insert_scheduled_interval(
        stream_intervals.at(
            static_cast<std::size_t>(
                best_stream
            )
        ),

        ScheduledInterval{
            best_start_time,
            best_finish_time,
            task_id
        }
    );


    result.task_stream[
        task_id
    ] = best_stream;


    result.task_start_time[
        task_id
    ] = best_start_time;


    result.task_finish_time[
        task_id
    ] = best_finish_time;


    result.makespan =
        std::max(
            result.makespan,
            best_finish_time
        );


    ++scheduled_task_count;


    /*
     * ==========================================================
     * 後続タスクのready状態を更新
     * ==========================================================
     */
    for (
        const int succ_id :
        importance.successors.at(
            task_id
        )
    ) {
      auto remaining_it =
          remaining_predecessor_count.find(
              succ_id
          );


      if (
          remaining_it ==
          remaining_predecessor_count.end()
      ) {
        throw std::runtime_error(
            "remaining predecessor information not found: " +
            std::to_string(succ_id)
        );
      }


      --remaining_it->second;


      if (
          remaining_it->second <
          0
      ) {
        throw std::runtime_error(
            "invalid predecessor count for task: " +
            std::to_string(succ_id)
        );
      }


      /*
       * 先行タスクがすべて配置済み
       */
      if (
          remaining_it->second ==
          0
      ) {
        ready_task_ids.push_back(
            succ_id
        );
      }
    }
  }


  /*
   * ============================================================
   * 全タスク配置確認
   * ============================================================
   */
  if (
      scheduled_task_count !=
      tasks.size()
  ) {
    throw std::runtime_error(
        "not all tasks were scheduled; "
        "dependency graph may contain a cycle"
    );
  }


  /*
   * タスク配置とStream内順序を固定した後でのみSM数を反映する。
   * Stage 4はこのSM-aware makespanを使って実行構成を比較する。
   */
  result.makespan =
      estimate_fixed_assignment_makespan(
          tasks,
          stream_sm_counts,
          result,
          reference_sm_count
      );


  return result;
}
