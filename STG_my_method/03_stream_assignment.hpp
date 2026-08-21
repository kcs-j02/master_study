#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "01_stg_analysis.hpp"
#include "02_task_importance.hpp"

/*
 * Stage 3: タスクをStreamへ配置した結果。
 */
struct StreamScheduleResult {
  std::unordered_map<int, int> task_stream;
  std::unordered_map<int, double> task_start_time;
  std::unordered_map<int, double> task_finish_time;
  double makespan = 0.0;
};

/*
 * Stream上に配置済みのタスク区間。
 */
struct ScheduledInterval {
  double start_time = 0.0;
  double finish_time = 0.0;
  int task_id = -1;
};

/*
 * dependency_ready_time以降でdurationを挿入できる、最初の空き時刻を返す。
 * intervalsは開始時刻順に整列済みであることを前提とする。
 */
inline double find_earliest_insertion_start(
    const std::vector<ScheduledInterval>& intervals,
    double dependency_ready_time,
    double duration
) {
  constexpr double epsilon = 1.0e-9;
  double candidate_start = dependency_ready_time;

  for (const auto& interval : intervals) {
    if (candidate_start + duration <= interval.start_time + epsilon) {
      return candidate_start;
    }

    if (candidate_start < interval.finish_time) {
      candidate_start = interval.finish_time;
    }
  }

  return candidate_start;
}

inline void insert_scheduled_interval(
    std::vector<ScheduledInterval>& intervals,
    ScheduledInterval interval
) {
  const auto position = std::lower_bound(
      intervals.begin(),
      intervals.end(),
      interval,
      [](const ScheduledInterval& lhs, const ScheduledInterval& rhs) {
        if (lhs.start_time != rhs.start_time) {
          return lhs.start_time < rhs.start_time;
        }
        return lhs.task_id < rhs.task_id;
      }
  );

  intervals.insert(position, interval);
}

/*
 * StreamごとのSM数を検証する。
 */
inline void validate_stream_sm_counts(
    const std::vector<int>& stream_sm_counts
) {
  if (stream_sm_counts.empty()) {
    throw std::invalid_argument(
        "stream_sm_counts must not be empty"
    );
  }

  for (std::size_t stream_id = 0;
       stream_id < stream_sm_counts.size();
       ++stream_id) {
    if (stream_sm_counts.at(stream_id) <= 0) {
      throw std::invalid_argument(
          "stream SM count must be positive: stream " +
          std::to_string(stream_id)
      );
    }
  }
}

namespace stream_assignment_detail {

/*
 * Stage 2の結果が、Stage 3へ渡された全タスクを覆っていることを確認する。
 * 重要度はここで再計算せず、欠落は入力エラーとして扱う。
 */
inline void validate_importance_result(
    const std::vector<TaskSpec>& tasks,
    const TaskImportanceResult& importance,
    const std::unordered_map<int, const TaskSpec*>& task_by_id
) {
  for (const auto& task : tasks) {
    if (importance.bottom_levels.find(task.id) ==
        importance.bottom_levels.end()) {
      throw std::invalid_argument(
          "bottom level not found for task: " +
          std::to_string(task.id)
      );
    }

    const auto successors_it = importance.successors.find(task.id);

    if (successors_it == importance.successors.end()) {
      throw std::invalid_argument(
          "successor information not found for task: " +
          std::to_string(task.id)
      );
    }

    for (const int successor_id : successors_it->second) {
      if (task_by_id.find(successor_id) == task_by_id.end()) {
        throw std::invalid_argument(
            "successor task not found: " +
            std::to_string(task.id) + " -> " +
            std::to_string(successor_id)
        );
      }
    }
  }
}

/*
 * 全先行タスクが完了する予測時刻を求める。
 */
inline double get_predecessor_ready_time(
    const TaskSpec& task,
    const std::unordered_map<int, double>& task_finish_time
) {
  double ready_time = 0.0;

  for (const int predecessor_id : task.preds) {
    const auto finish_it = task_finish_time.find(predecessor_id);

    if (finish_it == task_finish_time.end()) {
      throw std::runtime_error(
          "predecessor has not been scheduled: " +
          std::to_string(predecessor_id) + " -> " +
          std::to_string(task.id)
      );
    }

    ready_time = std::max(ready_time, finish_it->second);
  }

  return ready_time;
}

/*
 * 最も遅く終了する先行タスクのStreamを返す。
 */
inline int get_preferred_predecessor_stream(
    const TaskSpec& task,
    const std::unordered_map<int, double>& task_finish_time,
    const std::unordered_map<int, int>& task_stream
) {
  int preferred_stream = -1;
  double latest_finish = -1.0;

  for (const int predecessor_id : task.preds) {
    const auto finish_it = task_finish_time.find(predecessor_id);
    const auto stream_it = task_stream.find(predecessor_id);

    if (finish_it == task_finish_time.end() ||
        stream_it == task_stream.end()) {
      throw std::runtime_error(
          "predecessor information not found: " +
          std::to_string(predecessor_id)
      );
    }

    if (finish_it->second > latest_finish) {
      latest_finish = finish_it->second;
      preferred_stream = stream_it->second;
    }
  }

  return preferred_stream;
}

}  // namespace stream_assignment_detail

/*
 * Stage 2で求めた重要度を使い、依存関係を考慮したready-list配置を行う。
 *
 * - ready内ではbottom levelが大きいタスクを優先する。
 * - 各タスクは予測終了時刻が最小のStreamへ配置する。
 * - 依存待ちで生じたStream内の空き区間も利用する。
 *
 * reference_sm_countは候補間で共通の基準SM数である。0の場合は、
 * stream_sm_countsの合計を使用する。
 */
inline StreamScheduleResult place_tasks_on_streams(
    const std::vector<TaskSpec>& tasks,
    const TaskImportanceResult& importance,
    const std::vector<int>& stream_sm_counts,
    int reference_sm_count = 0
) {
  if (tasks.empty()) {
    throw std::invalid_argument("tasks must not be empty");
  }

  validate_stream_sm_counts(stream_sm_counts);

  if (reference_sm_count <= 0) {
    reference_sm_count = 0;
    for (const int sm_count : stream_sm_counts) {
      reference_sm_count += sm_count;
    }
  }

  if (reference_sm_count <= 0) {
    throw std::invalid_argument(
        "reference_sm_count must be positive"
    );
  }

  const auto task_by_id = make_task_table(tasks);
  stream_assignment_detail::validate_importance_result(
      tasks,
      importance,
      task_by_id
  );

  const int stream_count =
      static_cast<int>(stream_sm_counts.size());

  /* 最大SMを持つ最初のStreamを重要タスクの優先先とする。 */
  const int priority_stream_id = static_cast<int>(
      std::distance(
          stream_sm_counts.begin(),
          std::max_element(
              stream_sm_counts.begin(),
              stream_sm_counts.end()
          )
      )
  );

  std::vector<std::vector<ScheduledInterval>> stream_intervals(
      static_cast<std::size_t>(stream_count)
  );

  std::unordered_map<int, int> remaining_predecessor_count;
  remaining_predecessor_count.reserve(tasks.size());

  std::vector<int> ready_task_ids;
  ready_task_ids.reserve(tasks.size());

  for (const auto& task : tasks) {
    const int predecessor_count =
        static_cast<int>(task.preds.size());

    remaining_predecessor_count.emplace(
        task.id,
        predecessor_count
    );

    if (predecessor_count == 0) {
      ready_task_ids.push_back(task.id);
    }
  }

  StreamScheduleResult result;
  result.task_stream.reserve(tasks.size());
  result.task_start_time.reserve(tasks.size());
  result.task_finish_time.reserve(tasks.size());

  constexpr double epsilon = 1.0e-9;
  std::size_t scheduled_task_count = 0;

  while (!ready_task_ids.empty()) {
    /*
     * bottom levelが最大のready taskを選ぶ。
     * 同値なら処理時間が長いもの、さらに同値ならIDが小さいものを選ぶ。
     */
    std::size_t selected_ready_index = 0;

    for (std::size_t index = 1;
         index < ready_task_ids.size();
         ++index) {
      const int current_id = ready_task_ids.at(index);
      const int selected_id = ready_task_ids.at(selected_ready_index);
      const double current_rank =
          importance.bottom_levels.at(current_id);
      const double selected_rank =
          importance.bottom_levels.at(selected_id);

      bool select_current = false;

      if (current_rank > selected_rank + epsilon) {
        select_current = true;
      } else if (std::abs(current_rank - selected_rank) <= epsilon) {
        const auto current_proc_time =
            get_task_proc_time(*task_by_id.at(current_id));
        const auto selected_proc_time =
            get_task_proc_time(*task_by_id.at(selected_id));

        if (current_proc_time > selected_proc_time) {
          select_current = true;
        } else if (current_proc_time == selected_proc_time &&
                   current_id < selected_id) {
          select_current = true;
        }
      }

      if (select_current) {
        selected_ready_index = index;
      }
    }

    const int task_id = ready_task_ids.at(selected_ready_index);
    ready_task_ids.erase(
        ready_task_ids.begin() +
        static_cast<std::ptrdiff_t>(selected_ready_index)
    );

    const TaskSpec& task = *task_by_id.at(task_id);
    const double predecessor_ready_time =
        stream_assignment_detail::get_predecessor_ready_time(
            task,
            result.task_finish_time
        );
    const int preferred_stream =
        stream_assignment_detail::get_preferred_predecessor_stream(
            task,
            result.task_finish_time,
            result.task_stream
        );

    int best_stream = -1;
    double best_start_time = std::numeric_limits<double>::max();
    double best_finish_time = std::numeric_limits<double>::max();
    double priority_stream_start_time =
        std::numeric_limits<double>::max();
    double priority_stream_finish_time =
        std::numeric_limits<double>::max();

    for (int stream_id = 0; stream_id < stream_count; ++stream_id) {
      const int sm_count = stream_sm_counts.at(
          static_cast<std::size_t>(stream_id)
      );

      /* 小さいgridでは、タスクの並列度を超えるSMは使用できない。 */
      const int task_parallel_sm_limit =
          std::max(1, task.parallel_sm_limit);
      const int reference_effective_sm =
          std::min(reference_sm_count, task_parallel_sm_limit);
      const int stream_effective_sm =
          std::min(sm_count, task_parallel_sm_limit);
      const double estimated_proc_time =
          static_cast<double>(get_task_proc_time(task)) *
          static_cast<double>(reference_effective_sm) /
          static_cast<double>(stream_effective_sm);

      const double start_time = find_earliest_insertion_start(
          stream_intervals.at(static_cast<std::size_t>(stream_id)),
          predecessor_ready_time,
          estimated_proc_time
      );
      const double finish_time = start_time + estimated_proc_time;

      if (stream_id == priority_stream_id) {
        priority_stream_start_time = start_time;
        priority_stream_finish_time = finish_time;
      }

      bool select_current = false;

      if (finish_time < best_finish_time - epsilon) {
        select_current = true;
      } else if (std::abs(finish_time - best_finish_time) <= epsilon) {
        const bool current_is_preferred =
            stream_id == preferred_stream;
        const bool best_is_preferred =
            best_stream == preferred_stream;

        if (current_is_preferred && !best_is_preferred) {
          select_current = true;
        } else if (current_is_preferred == best_is_preferred) {
          if (start_time < best_start_time - epsilon) {
            select_current = true;
          } else if (std::abs(start_time - best_start_time) <= epsilon) {
            /* 同条件なら、大きいSMのStreamを温存する。 */
            const int best_sm =
                best_stream < 0
                    ? std::numeric_limits<int>::max()
                    : stream_sm_counts.at(
                          static_cast<std::size_t>(best_stream)
                      );

            if (sm_count < best_sm ||
                (sm_count == best_sm &&
                 (best_stream < 0 || stream_id < best_stream))) {
              select_current = true;
            }
          }
        }
      }

      if (select_current) {
        best_stream = stream_id;
        best_start_time = start_time;
        best_finish_time = finish_time;
      }
    }

    /*
     * 最速配置の部分makespanを悪化させない場合は、最大SMのStreamへ
     * タスクを集め、他Streamを後続の並列タスクへ残す。
     */
    const double best_partial_makespan =
        std::max(result.makespan, best_finish_time);

    if (priority_stream_finish_time <= best_partial_makespan + epsilon) {
      best_stream = priority_stream_id;
      best_start_time = priority_stream_start_time;
      best_finish_time = priority_stream_finish_time;
    }

    if (best_stream < 0) {
      throw std::runtime_error(
          "failed to select stream for task: " +
          std::to_string(task_id)
      );
    }

    insert_scheduled_interval(
        stream_intervals.at(static_cast<std::size_t>(best_stream)),
        ScheduledInterval{
            best_start_time,
            best_finish_time,
            task_id
        }
    );

    result.task_stream[task_id] = best_stream;
    result.task_start_time[task_id] = best_start_time;
    result.task_finish_time[task_id] = best_finish_time;
    result.makespan = std::max(result.makespan, best_finish_time);
    ++scheduled_task_count;

    for (const int successor_id : importance.successors.at(task_id)) {
      auto remaining_it =
          remaining_predecessor_count.find(successor_id);

      if (remaining_it == remaining_predecessor_count.end()) {
        throw std::runtime_error(
            "remaining predecessor information not found: " +
            std::to_string(successor_id)
        );
      }

      --remaining_it->second;

      if (remaining_it->second < 0) {
        throw std::runtime_error(
            "invalid predecessor count for task: " +
            std::to_string(successor_id)
        );
      }

      if (remaining_it->second == 0) {
        ready_task_ids.push_back(successor_id);
      }
    }
  }

  if (scheduled_task_count != tasks.size()) {
    throw std::runtime_error(
        "not all tasks were scheduled; dependency graph may contain a cycle"
    );
  }

  return result;
}
