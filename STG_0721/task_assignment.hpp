#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common_types.hpp"
#include "task_levelization.hpp"

/*
 * 最大レベル幅と上限値から使用するstream数を決める。
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

  return std::min(max_width, max_stream_count);
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
 * 全先行タスクが完了する予測時刻を求める。
 */
inline std::int64_t get_predecessor_ready_time(
    const TaskSpec& task,
    const std::unordered_map<int, std::int64_t>& task_finish_time
) {
  std::int64_t ready_time = 0;

  for (const int pred_id : task.preds) {
    const auto finish_it = task_finish_time.find(pred_id);

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
 *
 * 終了予測時刻が同じ候補streamが複数ある場合に、
 * 依存元と同じstreamを優先するために使う。
 */
inline int get_preferred_predecessor_stream(
    const TaskSpec& task,
    const std::unordered_map<int, std::int64_t>& task_finish_time,
    const std::unordered_map<int, int>& task_stream
) {
  int preferred_stream = -1;
  std::int64_t latest_finish = -1;

  for (const int pred_id : task.preds) {
    const auto finish_it = task_finish_time.find(pred_id);
    const auto stream_it = task_stream.find(pred_id);

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
 * stream割り当て規則
 *
 * 第1層・第2層以降を問わず、常に全streamを候補にする。
 *
 * 各streamについて、
 *
 *   start_time
 *     = max(
 *         stream_finish_time[stream_id],
 *         predecessor_ready_time
 *       )
 *
 *   finish_time
 *     = start_time + task.proc_time
 *
 * を計算し、finish_timeが最小になるstreamを選ぶ。
 *
 * これにより、先行タスクがstream 0に存在していても、
 * stream 1～stream_count-1を候補から除外しない。
 */
inline void assign_stream_to_node(
    const TaskLevels& levels,
    TaskDFG& dfg,
    const std::vector<TaskSpec>& tasks,
    int stream_count
) {
  if (stream_count <= 0) {
    throw std::invalid_argument(
        "stream_count must be positive"
    );
  }

  const auto task_by_id =
      make_task_table(tasks);

  /*
   * 各streamが次に空く予測時刻。
   *
   * 単なるproc_time合計ではなく、
   * 依存待ちによる空白時間も含んだ終了予測時刻を保持する。
   */
  std::vector<std::int64_t> stream_finish_time(
      static_cast<std::size_t>(stream_count),
      0
  );

  /*
   * 各タスクの終了予測時刻と割り当てstream。
   */
  std::unordered_map<int, std::int64_t> task_finish_time;
  std::unordered_map<int, int> task_stream;

  task_finish_time.reserve(tasks.size());
  task_stream.reserve(tasks.size());

  for (const auto& level : levels) {
    std::vector<int> ordered_task_ids = level;

    /*
     * 同一レベル内ではproc_timeが大きいタスクから配置する。
     */
    std::stable_sort(
        ordered_task_ids.begin(),
        ordered_task_ids.end(),
        [&](int lhs_id, int rhs_id) {
          const auto lhs_it = task_by_id.find(lhs_id);
          const auto rhs_it = task_by_id.find(rhs_id);

          if (lhs_it == task_by_id.end()) {
            throw std::runtime_error(
                "task specification not found: " +
                std::to_string(lhs_id)
            );
          }

          if (rhs_it == task_by_id.end()) {
            throw std::runtime_error(
                "task specification not found: " +
                std::to_string(rhs_id)
            );
          }

          return get_task_proc_time(*lhs_it->second) >
                 get_task_proc_time(*rhs_it->second);
        }
    );

    for (const int task_id : ordered_task_ids) {
      const auto task_it = task_by_id.find(task_id);

      if (task_it == task_by_id.end()) {
        throw std::runtime_error(
            "task specification not found: " +
            std::to_string(task_id)
        );
      }

      const auto node_it = dfg.nodes.find(task_id);

      if (node_it == dfg.nodes.end()) {
        throw std::runtime_error(
            "DFG node not found: " +
            std::to_string(task_id)
        );
      }

      const TaskSpec& task = *task_it->second;

      const std::int64_t task_proc_time =
          get_task_proc_time(task);

      const std::int64_t predecessor_ready_time =
          get_predecessor_ready_time(
              task,
              task_finish_time
          );

      const int preferred_stream =
          get_preferred_predecessor_stream(
              task,
              task_finish_time,
              task_stream
          );

      int best_stream = -1;

      std::int64_t best_start_time =
          std::numeric_limits<std::int64_t>::max();

      std::int64_t best_finish_time =
          std::numeric_limits<std::int64_t>::max();

      /*
       * 第2層以降も、stream_count本すべてを比較する。
       */
      for (int stream_id = 0;
           stream_id < stream_count;
           ++stream_id) {
        const std::int64_t stream_ready_time =
            stream_finish_time.at(
                static_cast<std::size_t>(stream_id)
            );

        const std::int64_t start_time =
            std::max(
                stream_ready_time,
                predecessor_ready_time
            );

        const std::int64_t finish_time =
            start_time + task_proc_time;

        bool select_current = false;

        if (finish_time < best_finish_time) {
          select_current = true;
        }
        else if (finish_time == best_finish_time) {
          /*
           * 終了予測時刻が同じなら、
           * 最も遅い依存元と同じstreamを優先する。
           */
          const bool current_is_preferred =
              (stream_id == preferred_stream);

          const bool best_is_preferred =
              (best_stream == preferred_stream);

          if (current_is_preferred && !best_is_preferred) {
            select_current = true;
          }
          else if (current_is_preferred == best_is_preferred) {
            /*
             * それでも同じなら、開始予測時刻が早い方を選ぶ。
             */
            if (start_time < best_start_time) {
              select_current = true;
            }
            else if (
                start_time == best_start_time &&
                (best_stream < 0 || stream_id < best_stream)
            ) {
              select_current = true;
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

      node_it->second.stream_id =
          best_stream;

      stream_finish_time.at(
          static_cast<std::size_t>(best_stream)
      ) = best_finish_time;

      task_finish_time[task_id] =
          best_finish_time;

      task_stream[task_id] =
          best_stream;
    }
  }
}

/*
 * stream数の決定と割り当てをまとめて実行する。
 */
inline int task_assignment(
    const TaskLevels& levels,
    TaskDFG& dfg,
    const std::vector<TaskSpec>& tasks,
    int max_stream_count
) {
  const int stream_count =
      decide_stream_count(
          levels,
          max_stream_count
      );

  assign_stream_to_node(
      levels,
      dfg,
      tasks,
      stream_count
  );

  return stream_count;
}