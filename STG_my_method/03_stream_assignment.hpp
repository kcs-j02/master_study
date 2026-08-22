#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
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
 * ready-listから予測bottom level最大のタスクを選択する。
 *
 * 各Streamについて
 *
 *   EFT(i,s)
 *
 *     = EST(i,s)
 *       + predicted_proc_time(i,s)
 *
 * を計算する。
 *
 *
 * predicted_proc_timeにはStage 2と同じ
 *
 *   estimate_task_proc_time_on_stream()
 *
 * を使用する。
 *
 *
 * そのため、
 *
 *   kTaskParallelSmLimit = 114
 *
 * であっても、
 *
 * 現在の予測モデルでは64 SMを超える領域に
 * 理想的な速度向上を仮定しない。
 * ================================================================
 */

struct StreamScheduleResult {
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
          "predicted bottom level not found for task: " +
          std::to_string(task.id)
      );
    }


    /*
     * 代表予測時間
     */
    if (
        importance.predicted_proc_times.find(
            task.id
        ) ==
        importance.predicted_proc_times.end()
    ) {
      throw std::invalid_argument(
          "predicted processing time not found for task: " +
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
 * Stage 2とStage 3のSM配分が同じか確認
 * ================================================================
 */
inline void
validate_same_sm_allocation(
    const TaskImportanceResult& importance,

    const std::vector<int>&
        stream_sm_counts
) {
  if (
      importance.stream_sm_counts !=
      stream_sm_counts
  ) {
    throw std::invalid_argument(
        "Stage 2 and Stage 3 use different SM allocations"
    );
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
 * Stage 3本体
 * ================================================================
 *
 * 1.
 *   ready-listから
 *   予測bottom level最大のタスクを選択
 *
 * 2.
 *   各StreamのESTを計算
 *
 * 3.
 *   Stage 2と同じモデルで
 *   予測実行時間を計算
 *
 * 4.
 *   EFT = EST + predicted_proc_time
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
 *   SM数が小さいStream
 *     ↓
 *   Stream ID
 *
 * の順に比較する。
 *
 *
 * 64 SMを超えて予測速度が飽和する場合、
 * 不必要に大きいSM Streamを選ばないようにする。
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


  stream_assignment_detail::
      validate_same_sm_allocation(
          importance,
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
     * 予測bottom level最大のタスクを選ぶ。
     *
     * 同値なら
     *
     *   代表予測実行時間
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
        const double current_time =
            importance.predicted_proc_times.at(
                current_id
            );


        const double selected_time =
            importance.predicted_proc_times.at(
                selected_id
            );


        /*
         * 代表予測時間が長い方
         */
        if (
            current_time >
            selected_time +
                epsilon
        ) {
          select_current =
              true;
        }


        /*
         * それも同じならID順
         */
        else if (
            std::abs(
                current_time -
                selected_time
            ) <= epsilon &&

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
      const int sm_count =
          stream_sm_counts.at(
              static_cast<std::size_t>(
                  stream_id
              )
          );


      /*
       * ========================================================
       * 予測実行時間
       *
       * Stage 2と同じモデル
       * ========================================================
       */
      const double predicted_proc_time =
          estimate_task_proc_time_on_stream(
              task,
              sm_count,
              reference_sm_count
          );


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
              predicted_proc_time
          );


      /*
       * EFT
       */
      const double finish_time =
          start_time +
          predicted_proc_time;


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
           * SM数が小さいStream
           *
           * 64 SMを超えて予測時間が同じなら、
           * 不必要に大きいSM Streamを選ばない。
           * ----------------------------------------------------
           */
          const int best_sm_count =
              best_stream < 0

                  ? std::numeric_limits<int>::max()

                  : stream_sm_counts.at(
                        static_cast<std::size_t>(
                            best_stream
                        )
                    );


          if (
              sm_count <
              best_sm_count
          ) {
            select_current =
                true;
          }


          /*
           * ----------------------------------------------------
           * 第4基準:
           *
           * SM数まで同じならStream ID
           * ----------------------------------------------------
           */
          else if (
              sm_count ==
                  best_sm_count &&

              (
                  best_stream < 0 ||
                  stream_id <
                      best_stream
              )
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


  return result;
}