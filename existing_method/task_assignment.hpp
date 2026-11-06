#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common_types.hpp"
#include "task_levelization.hpp"

struct StreamScheduleResult {
  std::unordered_map<int, int> task_stream;
  std::unordered_map<int, double> task_start_time;
  std::unordered_map<int, double> task_finish_time;
  double makespan = 0.0;
};

inline int decide_stream_count(
    const TaskLevels& levels,
    int max_stream_count
) {
  if (max_stream_count <= 0) {
    throw std::invalid_argument("max_stream_count must be positive");
  }
  int max_width = 0;
  for (const auto& level : levels) {
    max_width = std::max(max_width, static_cast<int>(level.size()));
  }
  if (max_width <= 0) {
    throw std::runtime_error("task levels are empty");
  }
  return std::min(max_width, max_stream_count);
}

inline std::unordered_map<int, const TaskSpec*> make_task_table(
    const std::vector<TaskSpec>& tasks
) {
  std::unordered_map<int, const TaskSpec*> task_by_id;
  task_by_id.reserve(tasks.size());
  for (const auto& task : tasks) {
    if (!task_by_id.emplace(task.id, &task).second) {
      throw std::runtime_error("duplicate task id: " + std::to_string(task.id));
    }
  }
  return task_by_id;
}

inline double existing_task_proc_time(const TaskSpec& task) {
  return static_cast<double>(std::max(0, task.proc_time));
}

inline std::unordered_map<int, std::vector<int>> make_successor_table(
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
      if (pred_id == task.id || task_by_id.find(pred_id) == task_by_id.end()) {
        throw std::runtime_error(
            "invalid predecessor: " + std::to_string(pred_id) +
            " -> " + std::to_string(task.id));
      }
      successors.at(pred_id).push_back(task.id);
    }
  }
  return successors;
}

inline double get_predecessor_ready_time(
    const TaskSpec& task,
    const std::unordered_map<int, double>& task_finish_time
) {
  double ready_time = 0.0;
  for (const int pred_id : task.preds) {
    const auto finish_it = task_finish_time.find(pred_id);
    if (finish_it == task_finish_time.end()) {
      throw std::runtime_error(
          "predecessor has not been scheduled: " +
          std::to_string(pred_id) + " -> " + std::to_string(task.id));
    }
    ready_time = std::max(ready_time, finish_it->second);
  }
  return ready_time;
}

inline int get_preferred_predecessor_stream(
    const TaskSpec& task,
    const std::unordered_map<int, int>& task_stream
) {
  if (task.preds.empty()) return -1;
  const int predecessor_id = *std::min_element(
      task.preds.begin(), task.preds.end());
  const auto stream_it = task_stream.find(predecessor_id);
  if (stream_it == task_stream.end()) {
    throw std::runtime_error(
        "predecessor stream not found: " + std::to_string(predecessor_id));
  }
  return stream_it->second;
}

/*
 * 既存手法のstream配置。
 *
 * - 入次数0のタスクをready-listへ入れ、タスクINDEX昇順で選ぶ。
 * - 依存元がないタスクはINDEX順にstreamへラウンドロビン配置する。
 * - 依存元があるタスクは依存元と同じstreamへ配置する。
 * - 複数の依存元が異なるstreamにある場合は、INDEXが最小の
 *   依存元が存在するstreamを使用する。
 * - proc_time、bottom level、予測終了時刻はstream選択に使わない。
 *
 * stream_sm_countsの値は配置には使わない。Existingは全通常streamで
 * GPUを共有し、Existing + GCもGC分割前に同じ配置を確定する。
 */
inline StreamScheduleResult simulate_stream_assignment(
    const std::vector<TaskSpec>& tasks,
    const std::vector<int>& stream_sm_counts,
    int reference_sm_count = 0
) {
  if (tasks.empty()) {
    throw std::invalid_argument("tasks must not be empty");
  }
  if (stream_sm_counts.empty()) {
    throw std::invalid_argument("stream_sm_counts must not be empty");
  }
  for (const int sm_count : stream_sm_counts) {
    if (sm_count <= 0) {
      throw std::invalid_argument("stream SM count must be positive");
    }
  }
  (void)reference_sm_count;

  const int stream_count = static_cast<int>(stream_sm_counts.size());
  const auto task_by_id = make_task_table(tasks);
  const auto successors = make_successor_table(tasks, task_by_id);

  std::unordered_map<int, int> remaining_predecessors;
  remaining_predecessors.reserve(tasks.size());
  std::vector<int> ready_task_ids;
  ready_task_ids.reserve(tasks.size());
  for (const auto& task : tasks) {
    const int count = static_cast<int>(task.preds.size());
    remaining_predecessors.emplace(task.id, count);
    if (count == 0) ready_task_ids.push_back(task.id);
  }

  std::vector<double> stream_finish_time(
      static_cast<std::size_t>(stream_count), 0.0);
  StreamScheduleResult result;
  result.task_stream.reserve(tasks.size());
  result.task_start_time.reserve(tasks.size());
  result.task_finish_time.reserve(tasks.size());

  std::size_t scheduled_count = 0;
  int next_root_stream = 0;
  while (!ready_task_ids.empty()) {
    std::sort(ready_task_ids.begin(), ready_task_ids.end());
    const int task_id = ready_task_ids.front();
    ready_task_ids.erase(ready_task_ids.begin());
    const TaskSpec& task = *task_by_id.at(task_id);

    int selected_stream = -1;
    if (task.preds.empty()) {
      selected_stream = next_root_stream;
      next_root_stream = (next_root_stream + 1) % stream_count;
    } else {
      selected_stream = get_preferred_predecessor_stream(
          task, result.task_stream);
    }

    const double predecessor_ready_time =
        get_predecessor_ready_time(task, result.task_finish_time);
    const double start = std::max(
        stream_finish_time.at(static_cast<std::size_t>(selected_stream)),
        predecessor_ready_time);
    const double finish = start + existing_task_proc_time(task);
    stream_finish_time.at(static_cast<std::size_t>(selected_stream)) = finish;
    result.task_stream[task_id] = selected_stream;
    result.task_start_time[task_id] = start;
    result.task_finish_time[task_id] = finish;
    result.makespan = std::max(result.makespan, finish);
    ++scheduled_count;

    for (const int successor_id : successors.at(task_id)) {
      int& count = remaining_predecessors.at(successor_id);
      --count;
      if (count < 0) {
        throw std::runtime_error(
            "invalid predecessor count: " + std::to_string(successor_id));
      }
      if (count == 0) ready_task_ids.push_back(successor_id);
    }
  }

  if (scheduled_count != tasks.size()) {
    throw std::runtime_error("cycle detected in task graph");
  }
  return result;
}

inline void apply_stream_schedule_to_dfg(
    TaskDFG& dfg,
    const StreamScheduleResult& schedule
) {
  for (const auto& [task_id, stream_id] : schedule.task_stream) {
    const auto node_it = dfg.nodes.find(task_id);
    if (node_it == dfg.nodes.end()) {
      throw std::runtime_error("DFG node not found: " + std::to_string(task_id));
    }
    node_it->second.stream_id = stream_id;
  }
}
