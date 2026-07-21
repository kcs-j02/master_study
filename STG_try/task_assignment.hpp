

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common_types.hpp"
#include "task_levelization.hpp"

/*
 * スケジューリング結果。
 */
struct StreamScheduleResult {
  std::unordered_map<int, int> task_stream;
  std::unordered_map<int, double> task_start_time;
  std::unordered_map<int, double> task_finish_time;
  double makespan = 0.0;
};

/*
 * stream上に配置済みのタスク区間。
 */
struct ScheduledInterval {
  double start_time = 0.0;
  double finish_time = 0.0;
  int task_id = -1;
};

/*
 * dependency_ready_time以降で、durationを挿入できる最初の空き時刻を返す。
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
    if (candidate_start + duration <=
        interval.start_time + epsilon) {
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
      [](const ScheduledInterval& lhs,
         const ScheduledInterval& rhs) {
        if (lhs.start_time != rhs.start_time) {
          return lhs.start_time < rhs.start_time;
        }
        return lhs.task_id < rhs.task_id;
      }
  );

  intervals.insert(position, interval);
}

/*
 * stream数とSM配分の選択結果。
 */
struct StreamConfigurationDecision {
  int stream_count = 0;
  std::vector<int> stream_sm_counts;
  double estimated_makespan =
      std::numeric_limits<double>::max();
};

/*
 * 最大レベル幅を取得する。
 */
inline int get_max_level_width(
    const TaskLevels& levels
) {
  int max_width = 0;

  for (const auto& level : levels) {
    max_width = std::max(
        max_width,
        static_cast<int>(level.size())
    );
  }

  if (max_width <= 0) {
    throw std::runtime_error(
        "task levels are empty"
    );
  }

  return max_width;
}

/*
 * 従来APIとの互換用。
 *
 * tasksとSM総数がないため、この関数単体では
 * makespanによる比較はできない。
 * 新規コードではdecide_stream_configurationを使用する。
 */
inline int decide_stream_count(
    const TaskLevels& levels,
    int max_stream_count
) {
  if (max_stream_count <= 0) {
    throw std::invalid_argument(
        "max_stream_count must be positive"
    );
  }

  return std::min(
      get_max_level_width(levels),
      max_stream_count
  );
}

/*
 * task_idからTaskSpecを取得する表を作る。
 */
inline std::unordered_map<int, const TaskSpec*> make_task_table(
    const std::vector<TaskSpec>& tasks
) {
  std::unordered_map<int, const TaskSpec*> task_by_id;
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
 * proc_timeが0以下のダミータスクは0として扱う。
 */
inline std::int64_t get_task_proc_time(
    const TaskSpec& task
) {
  return std::max<std::int64_t>(
      0,
      static_cast<std::int64_t>(task.proc_time)
  );
}

/*
 * predsから後続タスク表を作る。
 */
inline std::unordered_map<int, std::vector<int>>
make_successor_table(
    const std::vector<TaskSpec>& tasks,
    const std::unordered_map<int, const TaskSpec*>& task_by_id
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

      if (task_by_id.find(pred_id) == task_by_id.end()) {
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
 * task_idから終端までの最長処理時間を再帰的に計算する。
 *
 * bottom_level(task)
 *   = proc_time(task)
 *     + max(bottom_level(successor))
 */
inline double calculate_bottom_level_impl(
    int task_id,
    const std::unordered_map<int, const TaskSpec*>& task_by_id,
    const std::unordered_map<int, std::vector<int>>& successors,
    std::unordered_map<int, int>& visit_state,
    std::unordered_map<int, double>& bottom_level
) {
  const int state = visit_state[task_id];

  if (state == 2) {
    return bottom_level.at(task_id);
  }

  if (state == 1) {
    throw std::runtime_error(
        "cycle detected around task: " +
        std::to_string(task_id)
    );
  }

  visit_state[task_id] = 1;

  const auto task_it = task_by_id.find(task_id);

  if (task_it == task_by_id.end()) {
    throw std::runtime_error(
        "task specification not found: " +
        std::to_string(task_id)
    );
  }

  double longest_successor_path = 0.0;

  const auto successor_it = successors.find(task_id);

  if (successor_it == successors.end()) {
    throw std::runtime_error(
        "successor information not found: " +
        std::to_string(task_id)
    );
  }

  for (const int successor_id : successor_it->second) {
    longest_successor_path = std::max(
        longest_successor_path,
        calculate_bottom_level_impl(
            successor_id,
            task_by_id,
            successors,
            visit_state,
            bottom_level
        )
    );
  }

  const double rank =
      static_cast<double>(
          get_task_proc_time(*task_it->second)
      ) + longest_successor_path;

  bottom_level[task_id] = rank;
  visit_state[task_id] = 2;

  return rank;
}

/*
 * 全タスクのbottom levelを計算する。
 */
inline std::unordered_map<int, double> calculate_bottom_levels(
    const std::vector<TaskSpec>& tasks,
    const std::unordered_map<int, const TaskSpec*>& task_by_id,
    const std::unordered_map<int, std::vector<int>>& successors
) {
  std::unordered_map<int, int> visit_state;
  std::unordered_map<int, double> bottom_level;

  visit_state.reserve(tasks.size());
  bottom_level.reserve(tasks.size());

  for (const auto& task : tasks) {
    calculate_bottom_level_impl(
        task.id,
        task_by_id,
        successors,
        visit_state,
        bottom_level
    );
  }

  return bottom_level;
}

/*
 * 全先行タスクが完了する予測時刻を求める。
 */
inline double get_predecessor_ready_time(
    const TaskSpec& task,
    const std::unordered_map<int, double>& task_finish_time
) {
  double ready_time = 0.0;

  for (const int pred_id : task.preds) {
    const auto finish_it =
        task_finish_time.find(pred_id);

    if (finish_it == task_finish_time.end()) {
      throw std::runtime_error(
          "predecessor has not been scheduled: " +
          std::to_string(pred_id) +
          " -> " +
          std::to_string(task.id)
      );
    }

    ready_time = std::max(
        ready_time,
        finish_it->second
    );
  }

  return ready_time;
}

/*
 * 最も遅く終了する先行タスクのstreamを返す。
 */
inline int get_preferred_predecessor_stream(
    const TaskSpec& task,
    const std::unordered_map<int, double>& task_finish_time,
    const std::unordered_map<int, int>& task_stream
) {
  int preferred_stream = -1;
  double latest_finish = -1.0;

  for (const int pred_id : task.preds) {
    const auto finish_it =
        task_finish_time.find(pred_id);

    const auto stream_it =
        task_stream.find(pred_id);

    if (finish_it == task_finish_time.end() ||
        stream_it == task_stream.end()) {
      throw std::runtime_error(
          "predecessor information not found: " +
          std::to_string(pred_id)
      );
    }

    if (finish_it->second > latest_finish) {
      latest_finish = finish_it->second;
      preferred_stream = stream_it->second;
    }
  }

  return preferred_stream;
}

/*
 * stream SM数を検証する。
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

/*
 * 依存関係を考慮したready-list schedulingを行う。
 *
 * 変更点:
 *   1. 先行タスクが全て配置済みのタスクだけをreadyにする。
 *   2. ready内ではbottom levelが大きいタスクを優先する。
 *   3. 各タスクは予測終了時刻が最小のstreamへ割り当てる。
 *
 * reference_sm_countは候補間で共通の基準SM数。
 * stream数比較時に同じ値を使うことで、makespanを比較可能にする。
 */
inline StreamScheduleResult simulate_stream_assignment(
    const std::vector<TaskSpec>& tasks,
    const std::vector<int>& stream_sm_counts,
    int reference_sm_count = 0
) {
  if (tasks.empty()) {
    throw std::invalid_argument(
        "tasks must not be empty"
    );
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
  const auto successors =
      make_successor_table(tasks, task_by_id);
  const auto bottom_level =
      calculate_bottom_levels(
          tasks,
          task_by_id,
          successors
      );

  const int stream_count =
      static_cast<int>(stream_sm_counts.size());

  /*
   * 各streamに配置済みの実行区間。
   * 末尾だけでなく、依存待ちによって生じた空白へタスクを挿入する。
   */
  std::vector<std::vector<ScheduledInterval>> stream_intervals(
      static_cast<std::size_t>(stream_count)
  );

  std::unordered_map<int, int>
      remaining_predecessor_count;
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
     * 同値ならproc_timeが大きいもの、さらに同値ならIDが小さいもの。
     */
    std::size_t selected_ready_index = 0;

    for (std::size_t index = 1;
         index < ready_task_ids.size();
         ++index) {
      const int current_id = ready_task_ids.at(index);
      const int selected_id =
          ready_task_ids.at(selected_ready_index);

      const double current_rank =
          bottom_level.at(current_id);
      const double selected_rank =
          bottom_level.at(selected_id);

      bool select_current = false;

      if (current_rank > selected_rank + epsilon) {
        select_current = true;
      }
      else if (
          std::abs(current_rank - selected_rank) <= epsilon
      ) {
        const auto current_proc_time =
            get_task_proc_time(*task_by_id.at(current_id));
        const auto selected_proc_time =
            get_task_proc_time(*task_by_id.at(selected_id));

        if (current_proc_time > selected_proc_time) {
          select_current = true;
        }
        else if (
            current_proc_time == selected_proc_time &&
            current_id < selected_id
        ) {
          select_current = true;
        }
      }

      if (select_current) {
        selected_ready_index = index;
      }
    }

    const int task_id =
        ready_task_ids.at(selected_ready_index);

    ready_task_ids.erase(
        ready_task_ids.begin() +
        static_cast<std::ptrdiff_t>(selected_ready_index)
    );

    const TaskSpec& task = *task_by_id.at(task_id);

    const double predecessor_ready_time =
        get_predecessor_ready_time(
            task,
            result.task_finish_time
        );

    const int preferred_stream =
        get_preferred_predecessor_stream(
            task,
            result.task_finish_time,
            result.task_stream
        );

    int best_stream = -1;
    double best_start_time =
        std::numeric_limits<double>::max();
    double best_finish_time =
        std::numeric_limits<double>::max();

    for (int stream_id = 0;
         stream_id < stream_count;
         ++stream_id) {
      const int sm_count = stream_sm_counts.at(
          static_cast<std::size_t>(stream_id)
      );

      const double estimated_proc_time =
          static_cast<double>(get_task_proc_time(task)) *
          static_cast<double>(reference_sm_count) /
          static_cast<double>(sm_count);

      const double start_time =
          find_earliest_insertion_start(
              stream_intervals.at(
                  static_cast<std::size_t>(stream_id)
              ),
              predecessor_ready_time,
              estimated_proc_time
          );

      const double finish_time =
          start_time + estimated_proc_time;

      bool select_current = false;

      if (finish_time < best_finish_time - epsilon) {
        select_current = true;
      }
      else if (
          std::abs(finish_time - best_finish_time) <= epsilon
      ) {
        const bool current_is_preferred =
            stream_id == preferred_stream;
        const bool best_is_preferred =
            best_stream == preferred_stream;

        if (current_is_preferred && !best_is_preferred) {
          select_current = true;
        }
        else if (
            current_is_preferred == best_is_preferred
        ) {
          if (start_time < best_start_time - epsilon) {
            select_current = true;
          }
          else if (
              std::abs(start_time - best_start_time) <= epsilon
          ) {
            /*
             * 完了時刻も開始時刻も同じなら、
             * 大きいSMのstreamを温存するため小さいSMを優先する。
             */
            const int current_sm = sm_count;
            const int best_sm =
                best_stream < 0
                    ? std::numeric_limits<int>::max()
                    : stream_sm_counts.at(
                          static_cast<std::size_t>(best_stream)
                      );

            if (current_sm < best_sm) {
              select_current = true;
            }
            else if (
                current_sm == best_sm &&
                (best_stream < 0 || stream_id < best_stream)
            ) {
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

    if (best_stream < 0) {
      throw std::runtime_error(
          "failed to select stream for task: " +
          std::to_string(task_id)
      );
    }

    insert_scheduled_interval(
        stream_intervals.at(
            static_cast<std::size_t>(best_stream)
        ),
        ScheduledInterval{
            best_start_time,
            best_finish_time,
            task_id
        }
    );

    result.task_stream[task_id] = best_stream;
    result.task_start_time[task_id] = best_start_time;
    result.task_finish_time[task_id] = best_finish_time;
    result.makespan = std::max(
        result.makespan,
        best_finish_time
    );

    ++scheduled_task_count;

    for (const int successor_id : successors.at(task_id)) {
      auto remaining_it =
          remaining_predecessor_count.find(successor_id);

      if (remaining_it ==
          remaining_predecessor_count.end()) {
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

/*
 * 指定stream数に対して、SMを粒度単位で均等配分する。
 * 端数の単位は先頭streamから1単位ずつ追加する。
 */
inline std::vector<int> make_balanced_stream_sm_counts(
    int total_sm_count,
    int stream_count,
    int sm_granularity = 8,
    int min_sm_per_stream = 8
) {
  if (total_sm_count <= 0) {
    throw std::invalid_argument(
        "total_sm_count must be positive"
    );
  }

  if (stream_count <= 0) {
    throw std::invalid_argument(
        "stream_count must be positive"
    );
  }

  if (sm_granularity <= 0) {
    throw std::invalid_argument(
        "sm_granularity must be positive"
    );
  }

  if (min_sm_per_stream <= 0) {
    throw std::invalid_argument(
        "min_sm_per_stream must be positive"
    );
  }

  const int total_units =
      total_sm_count / sm_granularity;

  const int minimum_units =
      (min_sm_per_stream + sm_granularity - 1) /
      sm_granularity;

  if (total_units < stream_count * minimum_units) {
    throw std::runtime_error(
        "insufficient SMs for requested stream count"
    );
  }

  std::vector<int> stream_units(
      static_cast<std::size_t>(stream_count),
      minimum_units
  );

  int remaining_units =
      total_units - stream_count * minimum_units;

  const int common_extra =
      remaining_units / stream_count;
  const int remainder =
      remaining_units % stream_count;

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    stream_units.at(
        static_cast<std::size_t>(stream_id)
    ) += common_extra;

    if (stream_id < remainder) {
      ++stream_units.at(
          static_cast<std::size_t>(stream_id)
      );
    }
  }

  std::vector<int> stream_sm_counts;
  stream_sm_counts.reserve(
      static_cast<std::size_t>(stream_count)
  );

  for (const int units : stream_units) {
    stream_sm_counts.push_back(
        units * sm_granularity
    );
  }

  return stream_sm_counts;
}

/*
 * 実際に使用可能なSM配分候補を全て試し、
 * 予測makespanが最小の構成を選ぶ。
 *
 * Green Contextと通常streamを混在させる場合は、
 * main側で実際の候補を作り、この関数へ渡す。
 */
inline StreamConfigurationDecision decide_stream_configuration(
    const std::vector<TaskSpec>& tasks,
    const std::vector<std::vector<int>>&
        candidate_stream_sm_counts,
    int reference_sm_count = 0
) {
  if (candidate_stream_sm_counts.empty()) {
    throw std::invalid_argument(
        "candidate_stream_sm_counts must not be empty"
    );
  }

  if (reference_sm_count <= 0) {
    for (const auto& candidate :
         candidate_stream_sm_counts) {
      int candidate_total = 0;

      for (const int sm_count : candidate) {
        candidate_total += sm_count;
      }

      reference_sm_count = std::max(
          reference_sm_count,
          candidate_total
      );
    }
  }

  StreamConfigurationDecision best;
  constexpr double epsilon = 1.0e-9;

  for (const auto& candidate :
       candidate_stream_sm_counts) {
    validate_stream_sm_counts(candidate);

    const StreamScheduleResult schedule =
        simulate_stream_assignment(
            tasks,
            candidate,
            reference_sm_count
        );

    const int candidate_stream_count =
        static_cast<int>(candidate.size());

    bool select_current = false;

    if (schedule.makespan <
        best.estimated_makespan - epsilon) {
      select_current = true;
    }
    else if (
        std::abs(
            schedule.makespan -
            best.estimated_makespan
        ) <= epsilon &&
        (best.stream_count == 0 ||
         candidate_stream_count < best.stream_count)
    ) {
      /*
       * 同じ予測時間なら、管理コストを抑えるため
       * stream数が少ない構成を選ぶ。
       */
      select_current = true;
    }

    if (select_current) {
      best.stream_count = candidate_stream_count;
      best.stream_sm_counts = candidate;
      best.estimated_makespan = schedule.makespan;
    }
  }

  if (best.stream_count <= 0) {
    throw std::runtime_error(
        "failed to select stream configuration"
    );
  }

  return best;
}

/*
 * total_sm_countを1～上限streamへ均等分割した候補を作り、
 * 予測makespanが最小のstream数とSM配分を選ぶ。
 */
inline StreamConfigurationDecision decide_stream_configuration(
    const TaskLevels& levels,
    const std::vector<TaskSpec>& tasks,
    int max_stream_count,
    int total_sm_count,
    int sm_granularity = 8,
    int min_sm_per_stream = 8
) {
  if (max_stream_count <= 0) {
    throw std::invalid_argument(
        "max_stream_count must be positive"
    );
  }

  if (total_sm_count <= 0) {
    throw std::invalid_argument(
        "total_sm_count must be positive"
    );
  }

  if (sm_granularity <= 0) {
    throw std::invalid_argument(
        "sm_granularity must be positive"
    );
  }

  if (min_sm_per_stream <= 0) {
    throw std::invalid_argument(
        "min_sm_per_stream must be positive"
    );
  }

  const int total_units =
      total_sm_count / sm_granularity;

  const int minimum_units =
      (min_sm_per_stream + sm_granularity - 1) /
      sm_granularity;

  const int resource_limit =
      total_units / minimum_units;

  const int candidate_limit = std::min(
      std::min(
          get_max_level_width(levels),
          max_stream_count
      ),
      resource_limit
  );

  if (candidate_limit <= 0) {
    throw std::runtime_error(
        "no valid stream count candidate"
    );
  }

  std::vector<std::vector<int>> candidates;
  candidates.reserve(
      static_cast<std::size_t>(candidate_limit)
  );

  for (int stream_count = 1;
       stream_count <= candidate_limit;
       ++stream_count) {
    candidates.push_back(
        make_balanced_stream_sm_counts(
            total_sm_count,
            stream_count,
            sm_granularity,
            min_sm_per_stream
        )
    );
  }

  const int allocatable_sm_count =
      total_units * sm_granularity;

  return decide_stream_configuration(
      tasks,
      candidates,
      allocatable_sm_count
  );
}

/*
 * 計算済みスケジュールをDFGへ反映する。
 */
inline void apply_stream_schedule_to_dfg(
    TaskDFG& dfg,
    const StreamScheduleResult& schedule
) {
  for (const auto& [task_id, stream_id] :
       schedule.task_stream) {
    const auto node_it = dfg.nodes.find(task_id);

    if (node_it == dfg.nodes.end()) {
      throw std::runtime_error(
          "DFG node not found: " +
          std::to_string(task_id)
      );
    }

    node_it->second.stream_id = stream_id;
  }
}

/*
 * 依存関係、bottom level、stream内の空き区間を考慮して割り当てる。
 */
inline void assign_stream_to_node(
    const TaskLevels& levels,
    TaskDFG& dfg,
    const std::vector<TaskSpec>& tasks,
    const std::vector<int>& stream_sm_counts
) {
  (void)levels;

  const StreamScheduleResult schedule =
      simulate_stream_assignment(
          tasks,
          stream_sm_counts
      );

  apply_stream_schedule_to_dfg(dfg, schedule);
}

/*
 * stream_sm_countsを受け取り、SM数と依存関係を考慮して割り当てる。
 */
inline void task_assignment(
    const TaskLevels& levels,
    TaskDFG& dfg,
    const std::vector<TaskSpec>& tasks,
    const std::vector<int>& stream_sm_counts
) {
  assign_stream_to_node(
      levels,
      dfg,
      tasks,
      stream_sm_counts
  );
}