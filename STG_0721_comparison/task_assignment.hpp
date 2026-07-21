#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common_types.hpp"
#include "task_levelization.hpp"

/*
 * proc_timeを基準にしたStreamスケジューラ。
 *
 * 方針:
 *   1. proc_timeをタスクの基準実行時間として扱う。
 *   2. proc_timeが大きいタスクほど、多くのSMを必要とすると推定する。
 *   3. 各タスクを予測終了時刻が最小になるStreamへ割り当てる。
 *   4. 各Streamへ割り当てられたproc_time総量に比例してSMを再配分する。
 *   5. 1本から上限本数まで比較し、予測makespanが最小の構成を選ぶ。
 */

struct ScheduledInterval {
  double start_time = 0.0;
  double finish_time = 0.0;
  int task_id = -1;
};

struct StreamScheduleResult {
  std::unordered_map<int, int> task_stream;
  std::unordered_map<int, double> task_start_time;
  std::unordered_map<int, double> task_finish_time;

  std::vector<double> stream_proc_time_load;
  std::vector<double> stream_estimated_busy_time;

  double makespan = 0.0;
};

struct StreamConfigurationDecision {
  int stream_count = 0;
  std::vector<int> stream_sm_counts;
  double estimated_makespan =
      std::numeric_limits<double>::max();
};

struct ProcTimeSmModel {
  int available_sm = 0;
  int unit_sm = 0;
  int min_group_sm = 0;
  int max_stream_count = 0;

  int min_task_sm = 0;
  int max_task_sm = 0;
  int max_positive_proc_time = 0;
};

struct ProcTimeCandidateSummary {
  int stream_count = 0;
  std::vector<int> stream_sm_counts;
  double estimated_makespan = 0.0;
};

struct ProcTimeConfigurationResult {
  StreamConfigurationDecision decision;
  StreamScheduleResult schedule;

  double normal_only_makespan = 0.0;
  int evaluated_configuration_count = 0;

  ProcTimeSmModel model;
  std::vector<ProcTimeCandidateSummary> candidates;
};

inline int align_up_proc_time_int(
    int value,
    int alignment
) {
  if (alignment <= 0) {
    throw std::invalid_argument(
        "alignment must be positive"
    );
  }

  return
      ((value + alignment - 1) / alignment) *
      alignment;
}

inline int align_down_proc_time_int(
    int value,
    int alignment
) {
  if (alignment <= 0) {
    throw std::invalid_argument(
        "alignment must be positive"
    );
  }

  return (value / alignment) * alignment;
}

inline double get_task_proc_time_cost(
    const TaskSpec& task
) {
  return static_cast<double>(
      std::max(0, task.proc_time)
  );
}

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

inline std::unordered_map<int, const TaskSpec*>
make_task_table(
    const std::vector<TaskSpec>& tasks
) {
  std::unordered_map<int, const TaskSpec*> task_by_id;
  task_by_id.reserve(tasks.size());

  for (const auto& task : tasks) {
    const auto [iterator, inserted] =
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

inline std::unordered_map<int, std::vector<int>>
make_successor_table(
    const std::vector<TaskSpec>& tasks,
    const std::unordered_map<int, const TaskSpec*>& task_by_id
) {
  std::unordered_map<int, std::vector<int>> successors;
  successors.reserve(tasks.size());

  for (const auto& task : tasks) {
    successors.emplace(
        task.id,
        std::vector<int>{}
    );
  }

  for (const auto& task : tasks) {
    for (const int predecessor_id : task.preds) {
      if (predecessor_id == task.id) {
        throw std::runtime_error(
            "self dependency found: " +
            std::to_string(task.id)
        );
      }

      if (task_by_id.find(predecessor_id) ==
          task_by_id.end()) {
        throw std::runtime_error(
            "predecessor task not found: " +
            std::to_string(predecessor_id) +
            " -> " +
            std::to_string(task.id)
        );
      }

      successors.at(predecessor_id).push_back(
          task.id
      );
    }
  }

  return successors;
}

inline double calculate_proc_time_bottom_level_impl(
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

  double longest_successor_path = 0.0;

  for (const int successor_id :
       successors.at(task_id)) {
    longest_successor_path = std::max(
        longest_successor_path,
        calculate_proc_time_bottom_level_impl(
            successor_id,
            task_by_id,
            successors,
            visit_state,
            bottom_level
        )
    );
  }

  const double rank =
      get_task_proc_time_cost(
          *task_by_id.at(task_id)
      ) +
      longest_successor_path;

  bottom_level[task_id] = rank;
  visit_state[task_id] = 2;

  return rank;
}

inline std::unordered_map<int, double>
calculate_proc_time_bottom_levels(
    const std::vector<TaskSpec>& tasks,
    const std::unordered_map<int, const TaskSpec*>& task_by_id,
    const std::unordered_map<int, std::vector<int>>& successors
) {
  std::unordered_map<int, int> visit_state;
  std::unordered_map<int, double> bottom_level;

  visit_state.reserve(tasks.size());
  bottom_level.reserve(tasks.size());

  for (const auto& task : tasks) {
    calculate_proc_time_bottom_level_impl(
        task.id,
        task_by_id,
        successors,
        visit_state,
        bottom_level
    );
  }

  return bottom_level;
}

inline ProcTimeSmModel make_proc_time_sm_model(
    const std::vector<TaskSpec>& tasks,
    int available_sm,
    int unit_sm,
    int min_group_sm,
    int max_stream_count
) {
  if (tasks.empty()) {
    throw std::invalid_argument(
        "tasks must not be empty"
    );
  }

  if (available_sm <= 0 ||
      unit_sm <= 0 ||
      min_group_sm <= 0 ||
      max_stream_count <= 0) {
    throw std::invalid_argument(
        "invalid proc-time SM model parameter"
    );
  }

  int max_positive_proc_time = 0;

  for (const auto& task : tasks) {
    max_positive_proc_time = std::max(
        max_positive_proc_time,
        task.proc_time
    );
  }

  const int reserved_for_other_streams =
      std::max(0, max_stream_count - 1) *
      min_group_sm;

  int max_task_sm =
      available_sm - reserved_for_other_streams;

  max_task_sm = align_down_proc_time_int(
      max_task_sm,
      unit_sm
  );

  max_task_sm = std::max(
      min_group_sm,
      max_task_sm
  );

  max_task_sm = std::min(
      max_task_sm,
      available_sm
  );

  ProcTimeSmModel model;
  model.available_sm = available_sm;
  model.unit_sm = unit_sm;
  model.min_group_sm = min_group_sm;
  model.max_stream_count = max_stream_count;
  model.min_task_sm = min_group_sm;
  model.max_task_sm = max_task_sm;
  model.max_positive_proc_time =
      max_positive_proc_time;

  return model;
}

/*
 * proc_timeが大きいタスクほど大きいSM需要を持つと推定する。
 * sqrtを使い、極端に大きいproc_timeだけへSMが集中しすぎるのを防ぐ。
 */
inline int estimate_task_sm_demand(
    const TaskSpec& task,
    const ProcTimeSmModel& model
) {
  if (task.proc_time <= 0 ||
      model.max_positive_proc_time <= 0) {
    return model.min_task_sm;
  }

  const double normalized =
      static_cast<double>(task.proc_time) /
      static_cast<double>(
          model.max_positive_proc_time
      );

  const double scaled = std::sqrt(
      std::clamp(normalized, 0.0, 1.0)
  );

  const double raw_demand =
      static_cast<double>(model.min_task_sm) +
      scaled *
          static_cast<double>(
              model.max_task_sm -
              model.min_task_sm
          );

  int demand = align_up_proc_time_int(
      static_cast<int>(std::ceil(raw_demand)),
      model.unit_sm
  );

  demand = std::clamp(
      demand,
      model.min_task_sm,
      model.max_task_sm
  );

  return demand;
}

inline double estimate_task_duration(
    const TaskSpec& task,
    int allocated_sm,
    const ProcTimeSmModel& model
) {
  if (allocated_sm <= 0) {
    throw std::invalid_argument(
        "allocated_sm must be positive"
    );
  }

  const double base_proc_time =
      get_task_proc_time_cost(task);

  if (base_proc_time <= 0.0) {
    return 0.0;
  }

  const int demand_sm =
      estimate_task_sm_demand(task, model);

  const int effective_sm = std::max(
      1,
      std::min(allocated_sm, demand_sm)
  );

  return
      base_proc_time *
      static_cast<double>(demand_sm) /
      static_cast<double>(effective_sm);
}

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
    const ScheduledInterval& interval
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

inline double get_predecessor_ready_time(
    const TaskSpec& task,
    const std::unordered_map<int, double>& task_finish_time
) {
  double ready_time = 0.0;

  for (const int predecessor_id : task.preds) {
    const auto iterator =
        task_finish_time.find(predecessor_id);

    if (iterator == task_finish_time.end()) {
      throw std::runtime_error(
          "predecessor has not been scheduled: " +
          std::to_string(predecessor_id) +
          " -> " +
          std::to_string(task.id)
      );
    }

    ready_time = std::max(
        ready_time,
        iterator->second
    );
  }

  return ready_time;
}

inline int get_preferred_predecessor_stream(
    const TaskSpec& task,
    const std::unordered_map<int, double>& task_finish_time,
    const std::unordered_map<int, int>& task_stream
) {
  int preferred_stream = -1;
  double latest_finish_time = -1.0;

  for (const int predecessor_id : task.preds) {
    const double finish_time =
        task_finish_time.at(predecessor_id);

    if (finish_time > latest_finish_time) {
      latest_finish_time = finish_time;
      preferred_stream =
          task_stream.at(predecessor_id);
    }
  }

  return preferred_stream;
}

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

inline StreamScheduleResult simulate_proc_time_assignment(
    const std::vector<TaskSpec>& tasks,
    const std::vector<int>& stream_sm_counts,
    const ProcTimeSmModel& model
) {
  if (tasks.empty()) {
    throw std::invalid_argument(
        "tasks must not be empty"
    );
  }

  validate_stream_sm_counts(stream_sm_counts);

  const int stream_count =
      static_cast<int>(stream_sm_counts.size());

  const auto task_by_id = make_task_table(tasks);
  const auto successors =
      make_successor_table(tasks, task_by_id);
  const auto bottom_level =
      calculate_proc_time_bottom_levels(
          tasks,
          task_by_id,
          successors
      );

  std::vector<std::vector<ScheduledInterval>>
      stream_intervals(
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
  result.stream_proc_time_load.assign(
      static_cast<std::size_t>(stream_count),
      0.0
  );
  result.stream_estimated_busy_time.assign(
      static_cast<std::size_t>(stream_count),
      0.0
  );

  constexpr double epsilon = 1.0e-9;
  std::size_t scheduled_task_count = 0;

  while (!ready_task_ids.empty()) {
    std::size_t selected_ready_index = 0;

    for (std::size_t index = 1;
         index < ready_task_ids.size();
         ++index) {
      const int current_id =
          ready_task_ids.at(index);
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
          std::abs(current_rank - selected_rank) <=
          epsilon
      ) {
        const double current_proc_time =
            get_task_proc_time_cost(
                *task_by_id.at(current_id)
            );

        const double selected_proc_time =
            get_task_proc_time_cost(
                *task_by_id.at(selected_id)
            );

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
        static_cast<std::ptrdiff_t>(
            selected_ready_index
        )
    );

    const TaskSpec& task =
        *task_by_id.at(task_id);

    const double dependency_ready_time =
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

    const int task_demand_sm =
        estimate_task_sm_demand(task, model);

    int best_stream = -1;
    double best_start_time =
        std::numeric_limits<double>::max();
    double best_finish_time =
        std::numeric_limits<double>::max();
    int best_sm_waste =
        std::numeric_limits<int>::max();

    for (int stream_id = 0;
         stream_id < stream_count;
         ++stream_id) {
      const int allocated_sm =
          stream_sm_counts.at(
              static_cast<std::size_t>(stream_id)
          );

      const double duration =
          estimate_task_duration(
              task,
              allocated_sm,
              model
          );

      const double start_time =
          find_earliest_insertion_start(
              stream_intervals.at(
                  static_cast<std::size_t>(stream_id)
              ),
              dependency_ready_time,
              duration
          );

      const double finish_time =
          start_time + duration;

      const int sm_waste = std::abs(
          allocated_sm - task_demand_sm
      );

      bool select_current = false;

      if (finish_time < best_finish_time - epsilon) {
        select_current = true;
      }
      else if (
          std::abs(finish_time - best_finish_time) <=
          epsilon
      ) {
        const bool current_is_preferred =
            stream_id == preferred_stream;
        const bool best_is_preferred =
            best_stream == preferred_stream;

        if (current_is_preferred &&
            !best_is_preferred) {
          select_current = true;
        }
        else if (
            current_is_preferred == best_is_preferred
        ) {
          if (sm_waste < best_sm_waste) {
            select_current = true;
          }
          else if (sm_waste == best_sm_waste) {
            if (start_time < best_start_time - epsilon) {
              select_current = true;
            }
            else if (
                std::abs(
                    start_time - best_start_time
                ) <= epsilon &&
                (best_stream < 0 ||
                 stream_id < best_stream)
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
        best_sm_waste = sm_waste;
      }
    }

    if (best_stream < 0) {
      throw std::runtime_error(
          "failed to select stream for task: " +
          std::to_string(task_id)
      );
    }

    const double selected_duration =
        best_finish_time - best_start_time;

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
    result.task_start_time[task_id] =
        best_start_time;
    result.task_finish_time[task_id] =
        best_finish_time;

    result.stream_proc_time_load.at(
        static_cast<std::size_t>(best_stream)
    ) += get_task_proc_time_cost(task);

    result.stream_estimated_busy_time.at(
        static_cast<std::size_t>(best_stream)
    ) += selected_duration;

    result.makespan = std::max(
        result.makespan,
        best_finish_time
    );

    ++scheduled_task_count;

    for (const int successor_id :
         successors.at(task_id)) {
      int& remaining =
          remaining_predecessor_count.at(
              successor_id
          );

      --remaining;

      if (remaining < 0) {
        throw std::runtime_error(
            "invalid predecessor count for task " +
            std::to_string(successor_id)
        );
      }

      if (remaining == 0) {
        ready_task_ids.push_back(successor_id);
      }
    }
  }

  if (scheduled_task_count != tasks.size()) {
    throw std::runtime_error(
        "cycle detected while scheduling tasks"
    );
  }

  return result;
}

/*
 * 各Streamのproc_time総量に比例してSMを配分する。
 *
 * - stream 1以降はGreen Contextなのでunit_sm単位。
 * - 全Streamへmin_group_sm以上を確保する。
 * - available_smをunit_smへ丸めた余りはnormal stream 0へ付与する。
 */
inline std::vector<int> allocate_sm_by_proc_time_load(
    const std::vector<double>& stream_proc_time_load,
    int available_sm,
    int unit_sm,
    int min_group_sm
) {
  if (stream_proc_time_load.empty()) {
    throw std::invalid_argument(
        "stream_proc_time_load must not be empty"
    );
  }

  const int stream_count =
      static_cast<int>(
          stream_proc_time_load.size()
      );

  const int total_units =
      available_sm / unit_sm;

  const int minimum_units =
      align_up_proc_time_int(
          min_group_sm,
          unit_sm
      ) /
      unit_sm;

  if (total_units <
      stream_count * minimum_units) {
    throw std::runtime_error(
        "insufficient SMs for requested stream count"
    );
  }

  std::vector<int> stream_units(
      static_cast<std::size_t>(stream_count),
      minimum_units
  );

  int remaining_units =
      total_units -
      stream_count * minimum_units;

  double total_load = std::accumulate(
      stream_proc_time_load.begin(),
      stream_proc_time_load.end(),
      0.0
  );

  std::vector<double> normalized_load =
      stream_proc_time_load;

  if (total_load <= 0.0) {
    std::fill(
        normalized_load.begin(),
        normalized_load.end(),
        1.0
    );

    total_load = static_cast<double>(stream_count);
  }

  std::vector<double> fractional_remainder(
      static_cast<std::size_t>(stream_count),
      0.0
  );

  int assigned_extra_units = 0;

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    const double exact_extra =
        static_cast<double>(remaining_units) *
        normalized_load.at(
            static_cast<std::size_t>(stream_id)
        ) /
        total_load;

    const int whole_extra =
        static_cast<int>(std::floor(exact_extra));

    stream_units.at(
        static_cast<std::size_t>(stream_id)
    ) += whole_extra;

    assigned_extra_units += whole_extra;

    fractional_remainder.at(
        static_cast<std::size_t>(stream_id)
    ) =
        exact_extra -
        static_cast<double>(whole_extra);
  }

  int leftover_units =
      remaining_units - assigned_extra_units;

  std::vector<int> stream_order(
      static_cast<std::size_t>(stream_count)
  );

  std::iota(
      stream_order.begin(),
      stream_order.end(),
      0
  );

  std::sort(
      stream_order.begin(),
      stream_order.end(),
      [&](int lhs, int rhs) {
        const double lhs_fraction =
            fractional_remainder.at(
                static_cast<std::size_t>(lhs)
            );

        const double rhs_fraction =
            fractional_remainder.at(
                static_cast<std::size_t>(rhs)
            );

        if (lhs_fraction != rhs_fraction) {
          return lhs_fraction > rhs_fraction;
        }

        const double lhs_load =
            normalized_load.at(
                static_cast<std::size_t>(lhs)
            );

        const double rhs_load =
            normalized_load.at(
                static_cast<std::size_t>(rhs)
            );

        if (lhs_load != rhs_load) {
          return lhs_load > rhs_load;
        }

        return lhs < rhs;
      }
  );

  for (int index = 0;
       index < leftover_units;
       ++index) {
    const int stream_id =
        stream_order.at(
            static_cast<std::size_t>(
                index % stream_count
            )
        );

    ++stream_units.at(
        static_cast<std::size_t>(stream_id)
    );
  }

  std::vector<int> stream_sm_counts(
      static_cast<std::size_t>(stream_count),
      0
  );

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    stream_sm_counts.at(
        static_cast<std::size_t>(stream_id)
    ) =
        stream_units.at(
            static_cast<std::size_t>(stream_id)
        ) *
        unit_sm;
  }

  const int aligned_sm = total_units * unit_sm;
  const int remainder_sm =
      available_sm - aligned_sm;

  stream_sm_counts.at(0) += remainder_sm;

  return stream_sm_counts;
}

inline std::vector<int> make_initial_stream_sm_counts(
    int stream_count,
    int available_sm,
    int unit_sm,
    int min_group_sm
) {
  return allocate_sm_by_proc_time_load(
      std::vector<double>(
          static_cast<std::size_t>(stream_count),
          1.0
      ),
      available_sm,
      unit_sm,
      min_group_sm
  );
}

struct FixedStreamCountResult {
  std::vector<int> stream_sm_counts;
  StreamScheduleResult schedule;
};

inline FixedStreamCountResult optimize_fixed_stream_count(
    const std::vector<TaskSpec>& tasks,
    int stream_count,
    const ProcTimeSmModel& model,
    int max_iterations = 12
) {
  if (stream_count <= 0) {
    throw std::invalid_argument(
        "stream_count must be positive"
    );
  }

  if (max_iterations <= 0) {
    throw std::invalid_argument(
        "max_iterations must be positive"
    );
  }

  std::vector<int> current_sm_counts =
      make_initial_stream_sm_counts(
          stream_count,
          model.available_sm,
          model.unit_sm,
          model.min_group_sm
      );

  FixedStreamCountResult best;
  best.schedule.makespan =
      std::numeric_limits<double>::max();

  std::vector<std::vector<int>> visited;

  for (int iteration = 0;
       iteration < max_iterations;
       ++iteration) {
    if (std::find(
            visited.begin(),
            visited.end(),
            current_sm_counts
        ) != visited.end()) {
      break;
    }

    visited.push_back(current_sm_counts);

    const StreamScheduleResult schedule =
        simulate_proc_time_assignment(
            tasks,
            current_sm_counts,
            model
        );

    if (schedule.makespan <
        best.schedule.makespan) {
      best.stream_sm_counts = current_sm_counts;
      best.schedule = schedule;
    }

    const std::vector<int> next_sm_counts =
        allocate_sm_by_proc_time_load(
            schedule.stream_proc_time_load,
            model.available_sm,
            model.unit_sm,
            model.min_group_sm
        );

    if (next_sm_counts == current_sm_counts) {
      break;
    }

    current_sm_counts = next_sm_counts;
  }

  if (best.stream_sm_counts.empty()) {
    throw std::runtime_error(
        "failed to optimize fixed stream count"
    );
  }

  return best;
}

inline ProcTimeConfigurationResult
decide_proc_time_stream_configuration(
    const std::vector<TaskSpec>& tasks,
    const TaskLevels& levels,
    int max_stream_count,
    int available_sm,
    int unit_sm,
    int min_group_sm
) {
  if (tasks.empty()) {
    throw std::invalid_argument(
        "tasks must not be empty"
    );
  }

  const int total_units =
      available_sm / unit_sm;

  const int minimum_units =
      align_up_proc_time_int(
          min_group_sm,
          unit_sm
      ) /
      unit_sm;

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
        "no valid stream-count candidate"
    );
  }

  const ProcTimeSmModel model =
      make_proc_time_sm_model(
          tasks,
          available_sm,
          unit_sm,
          min_group_sm,
          candidate_limit
      );

  ProcTimeConfigurationResult result;
  result.model = model;
  result.candidates.reserve(
      static_cast<std::size_t>(candidate_limit)
  );

  constexpr double epsilon = 1.0e-9;

  for (int stream_count = 1;
       stream_count <= candidate_limit;
       ++stream_count) {
    const FixedStreamCountResult fixed_result =
        optimize_fixed_stream_count(
            tasks,
            stream_count,
            model
        );

    ProcTimeCandidateSummary summary;
    summary.stream_count = stream_count;
    summary.stream_sm_counts =
        fixed_result.stream_sm_counts;
    summary.estimated_makespan =
        fixed_result.schedule.makespan;

    result.candidates.push_back(summary);
    ++result.evaluated_configuration_count;

    if (stream_count == 1) {
      result.normal_only_makespan =
          fixed_result.schedule.makespan;
    }

    bool select_current = false;

    if (fixed_result.schedule.makespan <
        result.decision.estimated_makespan -
        epsilon) {
      select_current = true;
    }
    else if (
        std::abs(
            fixed_result.schedule.makespan -
            result.decision.estimated_makespan
        ) <= epsilon &&
        (result.decision.stream_count == 0 ||
         stream_count <
             result.decision.stream_count)
    ) {
      select_current = true;
    }

    if (select_current) {
      result.decision.stream_count =
          stream_count;
      result.decision.stream_sm_counts =
          fixed_result.stream_sm_counts;
      result.decision.estimated_makespan =
          fixed_result.schedule.makespan;
      result.schedule = fixed_result.schedule;
    }
  }

  if (result.decision.stream_count <= 0) {
    throw std::runtime_error(
        "failed to select proc-time stream configuration"
    );
  }

  return result;
}

/*
 * 既存コードとの互換用。
 * この関数もproc_timeモデルで割り当てる。
 */
inline StreamScheduleResult simulate_stream_assignment(
    const std::vector<TaskSpec>& tasks,
    const std::vector<int>& stream_sm_counts,
    int reference_sm_count = 0
) {
  if (reference_sm_count <= 0) {
    reference_sm_count = std::accumulate(
        stream_sm_counts.begin(),
        stream_sm_counts.end(),
        0
    );
  }

  const int stream_count =
      static_cast<int>(stream_sm_counts.size());

  const int unit_sm = 8;
  const int min_group_sm = 8;

  const ProcTimeSmModel model =
      make_proc_time_sm_model(
          tasks,
          reference_sm_count,
          unit_sm,
          min_group_sm,
          std::max(1, stream_count)
      );

  return simulate_proc_time_assignment(
      tasks,
      stream_sm_counts,
      model
  );
}

inline void apply_stream_schedule_to_dfg(
    TaskDFG& dfg,
    const StreamScheduleResult& schedule
) {
  for (const auto& [task_id, stream_id] :
       schedule.task_stream) {
    const auto node_iterator =
        dfg.nodes.find(task_id);

    if (node_iterator == dfg.nodes.end()) {
      throw std::runtime_error(
          "DFG node not found: " +
          std::to_string(task_id)
      );
    }

    node_iterator->second.stream_id = stream_id;
  }
}

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

  apply_stream_schedule_to_dfg(
      dfg,
      schedule
  );
}

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