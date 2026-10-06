#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "00_pipeline_configuration.hpp"
#include "03_stream_assignment.hpp"

/*
 * Stage 4: Stage 3で確定したタスク配置にSMを配分する。
 *
 * 各Stream数の配置は変更せず、SM配分後のStream負荷を評価する。
 * デフォルト構成では8 SM単位の配分を全探索し、明示指定された
 * 構成では指定SM数をそのまま評価する。最後に予測makespanが
 * 最小の構成を選択する。
 */

/*
 * SM数を考慮したタスクの予測実行時間。
 *
 * proc_timeの基準と1タスクのSM上限は実測に基づき64 SMとする。
 * 64 SM未満では実行カーネルと同じblock/thread構成から、1 threadが
 * 担当する要素数（処理pass数）を求める。
 */
inline double estimate_task_proc_time_on_stream(
    const TaskSpec& task,
    int stream_sm_count,
    int reference_sm_count = kSchedulingReferenceSmCount
) {
  if (stream_sm_count <= 0) {
    throw std::invalid_argument(
        "stream_sm_count must be positive"
    );
  }

  if (reference_sm_count <= 0) {
    throw std::invalid_argument(
        "reference_sm_count must be positive"
    );
  }

  if (kProcTimeReferenceSmCount <= 0) {
    throw std::logic_error(
        "kProcTimeReferenceSmCount must be positive"
    );
  }

  const double base_proc_time =
      static_cast<double>(get_task_proc_time(task));

  if (base_proc_time <= 0.0) {
    return 0.0;
  }

  const int task_sm_limit =
      std::max(1, task.parallel_sm_limit);

  const int base_effective_sm =
      std::min(kProcTimeReferenceSmCount, task_sm_limit);

  const int stream_effective_sm =
      std::max(
          1,
          std::min({
              stream_sm_count,
              reference_sm_count,
              task_sm_limit
          })
      );

  const auto calculate_pass_count = [](int effective_sm) {
    constexpr int warp_size = 32;
    constexpr int maximum_threads_per_block = 256;

    const int elements_per_block =
        (kTaskElementCount + effective_sm - 1) / effective_sm;
    const int warp_aligned_threads =
        ((elements_per_block + warp_size - 1) / warp_size) * warp_size;
    const int threads_per_block = std::max(
        warp_size,
        std::min(maximum_threads_per_block, warp_aligned_threads)
    );

    return
        (elements_per_block + threads_per_block - 1) /
        threads_per_block;
  };

  const int base_pass_count = calculate_pass_count(base_effective_sm);
  const int stream_pass_count = calculate_pass_count(stream_effective_sm);

  return
      base_proc_time *
      static_cast<double>(stream_pass_count) /
      static_cast<double>(base_pass_count);
}

inline int initial_sm_per_stream_for_count(int stream_count) {
  switch (stream_count) {
    case 1: return 114;
    case 2: return 56;
    case 3: return 32;
    case 4: return 24;
    case 5: return 16;
    default:
      throw std::invalid_argument(
          "stream_count must be between 1 and 5"
      );
  }
}

inline double estimate_stream_processing_time_sum(
    const std::vector<TaskSpec>& tasks,
    const StreamScheduleResult& placement,
    const std::vector<int>& stream_sm_counts,
    int target_stream_id
) {
  validate_stream_sm_counts(stream_sm_counts);

  if (placement.stream_count !=
      static_cast<int>(stream_sm_counts.size())) {
    throw std::invalid_argument(
        "placement stream count and SM allocation do not match"
    );
  }

  if (target_stream_id < 0 ||
      target_stream_id >= placement.stream_count) {
    throw std::out_of_range("target stream id is out of range");
  }

  double processing_time = 0.0;

  for (const auto& task : tasks) {
    const int stream_id = placement.task_stream.at(task.id);

    if (stream_id < 0 || stream_id >= placement.stream_count) {
      throw std::invalid_argument(
          "task has an invalid stream assignment: " +
          std::to_string(task.id)
      );
    }

    if (stream_id != target_stream_id) {
      continue;
    }

    processing_time += estimate_task_proc_time_on_stream(
        task,
        stream_sm_counts.at(static_cast<std::size_t>(stream_id)),
        kSchedulingReferenceSmCount
    );
  }

  return processing_time;
}

inline double estimate_stream_load_makespan(
    const std::vector<TaskSpec>& tasks,
    const StreamScheduleResult& placement,
    const std::vector<int>& stream_sm_counts
) {
  validate_stream_sm_counts(stream_sm_counts);

  if (placement.stream_count !=
      static_cast<int>(stream_sm_counts.size())) {
    throw std::invalid_argument(
        "placement stream count and SM allocation do not match"
    );
  }

  double makespan = 0.0;

  for (std::size_t stream_id = 0;
       stream_id < stream_sm_counts.size();
       ++stream_id) {
    makespan = std::max(
        makespan,
        estimate_stream_processing_time_sum(
            tasks,
            placement,
            stream_sm_counts,
            static_cast<int>(stream_id)
        )
    );
  }

  return makespan;
}

struct StreamSmAllocationResult {
  std::vector<int> initial_pool_optimized_sm_counts;
  std::vector<int> after_stream_0_remainder_sm_counts;
  std::vector<int> runtime_sm_counts;
  double initial_pool_makespan = 0.0;
  double final_makespan = 0.0;
};

inline StreamSmAllocationResult optimize_stream_sm_allocation(
    const std::vector<TaskSpec>& tasks,
    const StreamScheduleResult& placement
) {
  constexpr int allocation_unit_sm = 8;
  const int stream_count = placement.stream_count;
  const int initial_sm_per_stream =
      initial_sm_per_stream_for_count(stream_count);
  const int initial_pool_sm =
      stream_count * initial_sm_per_stream;
  constexpr int stream_0_remainder_sm = 2;
  const int remaining_extra_sm =
      kSchedulingReferenceSmCount -
      initial_pool_sm -
      stream_0_remainder_sm;

  if (stream_count < 2 ||
      initial_pool_sm % allocation_unit_sm != 0 ||
      remaining_extra_sm < 0 ||
      remaining_extra_sm % allocation_unit_sm != 0) {
    throw std::invalid_argument(
        "invalid stream count for SM allocation optimization"
    );
  }

  /*
   * base_sm_countsを減らさず、追加する8 SMチャンクの全配分を探索する。
   * 評価値は各StreamのSM補正後処理時間合計の最大値とする。
   */
  auto optimize_additional_chunks = [
      &tasks,
      &placement,
      stream_count
  ](
      const std::vector<int>& base_sm_counts,
      int additional_chunk_count
  ) {
    std::vector<int> best_sm_counts;
    double best_makespan = std::numeric_limits<double>::max();
    long long best_addition_deviation =
        std::numeric_limits<long long>::max();
    constexpr double epsilon = 1.0e-9;
    std::vector<int> additions(
        static_cast<std::size_t>(stream_count),
        0
    );

    std::function<void(int, int)> search = [
        &
    ](int stream_id, int remaining_chunks) {
      if (stream_id == stream_count - 1) {
        additions.at(static_cast<std::size_t>(stream_id)) =
            remaining_chunks;

        std::vector<int> candidate = base_sm_counts;
        long long addition_deviation = 0;

        for (int id = 0; id < stream_count; ++id) {
          const int addition =
              additions.at(static_cast<std::size_t>(id));
          candidate.at(static_cast<std::size_t>(id)) +=
              addition * allocation_unit_sm;
          addition_deviation +=
              static_cast<long long>(addition) * addition;
        }

        const double makespan = estimate_stream_load_makespan(
            tasks,
            placement,
            candidate
        );

        if (makespan < best_makespan - epsilon ||
            (std::abs(makespan - best_makespan) <= epsilon &&
             addition_deviation < best_addition_deviation)) {
          best_sm_counts = std::move(candidate);
          best_makespan = makespan;
          best_addition_deviation = addition_deviation;
        }
        return;
      }

      for (int chunks = 0;
           chunks <= remaining_chunks;
           ++chunks) {
        additions.at(static_cast<std::size_t>(stream_id)) = chunks;
        search(stream_id + 1, remaining_chunks - chunks);
      }
    };

    search(0, additional_chunk_count);

    if (best_sm_counts.empty()) {
      throw std::runtime_error(
          "failed to optimize additional SM chunks"
      );
    }

    return std::pair<std::vector<int>, double>{
        std::move(best_sm_counts),
        best_makespan
    };
  };

  /* 第1段階: 初期総量だけを全探索する（各Stream最低8 SM）。 */
  std::vector<int> minimum_sm_counts(
      static_cast<std::size_t>(stream_count),
      allocation_unit_sm
  );
  const int initial_additional_chunks =
      initial_pool_sm / allocation_unit_sm - stream_count;
  auto initial_optimization = optimize_additional_chunks(
      minimum_sm_counts,
      initial_additional_chunks
  );

  /* 第1段階の結果を固定し、Stream 0へ端数2 SMを追加する。 */
  std::vector<int> after_stream_0_remainder =
      initial_optimization.first;
  after_stream_0_remainder.front() += stream_0_remainder_sm;

  /* 第2段階: 残った余剰SMだけを追加する全配分を探索する。 */
  const int remaining_extra_chunks =
      remaining_extra_sm / allocation_unit_sm;
  auto final_optimization = optimize_additional_chunks(
      after_stream_0_remainder,
      remaining_extra_chunks
  );

  StreamSmAllocationResult result;
  result.initial_pool_optimized_sm_counts =
      std::move(initial_optimization.first);
  result.after_stream_0_remainder_sm_counts =
      std::move(after_stream_0_remainder);
  result.runtime_sm_counts =
      std::move(final_optimization.first);
  result.initial_pool_makespan = initial_optimization.second;
  result.final_makespan = final_optimization.second;

  return result;
}

struct SmAllocationCandidate {
  std::vector<int> sm_counts;
  StreamScheduleResult schedule;
  double estimated_makespan = std::numeric_limits<double>::max();
};

struct SmAllocationDecision {
  std::vector<int> sm_counts;
  StreamScheduleResult schedule;
  double estimated_makespan = std::numeric_limits<double>::max();
};

inline void validate_sm_allocation_candidate(
    const SmAllocationCandidate& candidate
) {
  validate_stream_sm_counts(candidate.sm_counts);

  if (!std::isfinite(candidate.estimated_makespan) ||
      candidate.estimated_makespan < 0.0) {
    throw std::invalid_argument(
        "candidate makespan must be finite and non-negative"
    );
  }

  if (candidate.schedule.stream_count !=
      static_cast<int>(candidate.sm_counts.size())) {
    throw std::invalid_argument(
        "candidate schedule and SM allocation do not match"
    );
  }
}

/*
 * 同じStream数に複数候補がある場合、その中で最小makespanを残す。
 */
inline std::vector<SmAllocationCandidate>
select_best_candidate_per_stream_count(
    const std::vector<SmAllocationCandidate>& candidates
) {
  if (candidates.empty()) {
    throw std::invalid_argument("candidates must not be empty");
  }

  constexpr double epsilon = 1.0e-9;
  std::vector<SmAllocationCandidate> best_candidates;

  for (const auto& candidate : candidates) {
    validate_sm_allocation_candidate(candidate);
  }

  for (int stream_count = 1;
       stream_count <= kMaximumConfiguredStreamCount;
       ++stream_count) {
    const SmAllocationCandidate* best = nullptr;

    for (const auto& candidate : candidates) {
      if (static_cast<int>(candidate.sm_counts.size()) != stream_count) {
        continue;
      }

      if (best == nullptr ||
          candidate.estimated_makespan <
              best->estimated_makespan - epsilon) {
        best = &candidate;
      }
    }

    if (best != nullptr) {
      best_candidates.push_back(*best);
    }
  }

  if (best_candidates.empty()) {
    throw std::runtime_error(
        "no valid candidate was found for any stream count"
    );
  }

  return best_candidates;
}

/*
 * 全候補から予測makespan最小の実行構成を選ぶ。
 * 同値ならStream数が少ない候補を選ぶ。
 */
inline SmAllocationDecision compare_sm_allocation_candidates(
    const std::vector<SmAllocationCandidate>& candidates
) {
  if (candidates.empty()) {
    throw std::invalid_argument("candidates must not be empty");
  }

  constexpr double epsilon = 1.0e-9;
  std::size_t best_index = 0;

  validate_sm_allocation_candidate(candidates.front());

  for (std::size_t index = 1; index < candidates.size(); ++index) {
    validate_sm_allocation_candidate(candidates[index]);

    const auto& candidate = candidates[index];
    const auto& best = candidates[best_index];
    const double difference =
        candidate.estimated_makespan - best.estimated_makespan;

    if (difference < -epsilon ||
        (std::abs(difference) <= epsilon &&
         candidate.sm_counts.size() < best.sm_counts.size())) {
      best_index = index;
    }
  }

  const auto& best = candidates[best_index];

  return {
      best.sm_counts,
      best.schedule,
      best.estimated_makespan
  };
}

/*
 * 1つのStage 3配置に対し、要求されたSM構成を評価する。
 * デフォルト最適化時は要求値のStream数だけを使い、最終SM配分を探索する。
 */
inline SmAllocationCandidate allocate_sms_for_placement(
    const std::vector<TaskSpec>& tasks,
    const StreamScheduleResult& placement,
    const std::vector<int>& requested_sm_counts,
    bool optimize_default
) {
  validate_stream_sm_counts(requested_sm_counts);

  if (placement.stream_count !=
      static_cast<int>(requested_sm_counts.size())) {
    throw std::invalid_argument(
        "placement stream count and requested SM allocation do not match"
    );
  }

  std::vector<int> allocated_sm_counts = requested_sm_counts;
  double estimated_makespan = estimate_stream_load_makespan(
      tasks,
      placement,
      allocated_sm_counts
  );

  if (optimize_default && placement.stream_count >= 2) {
    StreamSmAllocationResult optimized =
        optimize_stream_sm_allocation(tasks, placement);
    allocated_sm_counts = std::move(optimized.runtime_sm_counts);
    estimated_makespan = optimized.final_makespan;
  }

  return {
      std::move(allocated_sm_counts),
      placement,
      estimated_makespan
  };
}

struct SmAllocationStageResult {
  std::vector<SmAllocationCandidate> all_candidates;
  std::vector<SmAllocationCandidate> best_per_stream_count;
  SmAllocationDecision decision;
};

/*
 * Stage 4本体。
 *
 * Stage 3はStream数ごとに1つの配置を作る。SM候補側に同じStream数の
 * 候補が複数あれば、その配置を再利用して各候補を評価する。
 */
inline SmAllocationStageResult allocate_sm_resources(
    const std::vector<TaskSpec>& tasks,
    const std::vector<StreamScheduleResult>& placements,
    const std::vector<std::vector<int>>& requested_sm_count_candidates,
    bool optimize_default
) {
  if (placements.empty()) {
    throw std::invalid_argument("placements must not be empty");
  }

  if (requested_sm_count_candidates.empty()) {
    throw std::invalid_argument(
        "requested SM count candidates must not be empty"
    );
  }

  std::vector<const StreamScheduleResult*> placement_by_stream_count(
      static_cast<std::size_t>(kMaximumConfiguredStreamCount + 1),
      nullptr
  );

  for (const auto& placement : placements) {
    if (placement.stream_count < 1 ||
        placement.stream_count > kMaximumConfiguredStreamCount) {
      throw std::invalid_argument(
          "placement stream count must be between 1 and " +
          std::to_string(kMaximumConfiguredStreamCount)
      );
    }

    auto& stored_placement = placement_by_stream_count.at(
        static_cast<std::size_t>(placement.stream_count)
    );

    if (stored_placement != nullptr) {
      throw std::invalid_argument(
          "duplicate placement for stream count: " +
          std::to_string(placement.stream_count)
      );
    }

    stored_placement = &placement;
  }

  SmAllocationStageResult result;
  result.all_candidates.reserve(requested_sm_count_candidates.size());

  for (const auto& requested_sm_counts :
       requested_sm_count_candidates) {
    validate_stream_sm_counts(requested_sm_counts);

    const int stream_count =
        static_cast<int>(requested_sm_counts.size());

    if (stream_count > kMaximumConfiguredStreamCount) {
      throw std::invalid_argument(
          "requested SM allocation exceeds the maximum stream count"
      );
    }

    const StreamScheduleResult* placement =
        placement_by_stream_count.at(
            static_cast<std::size_t>(stream_count)
        );

    if (placement == nullptr) {
      throw std::invalid_argument(
          "missing placement for stream count: " +
          std::to_string(stream_count)
      );
    }

    result.all_candidates.push_back(
        allocate_sms_for_placement(
            tasks,
            *placement,
            requested_sm_counts,
            optimize_default
        )
    );
  }

  result.best_per_stream_count =
      select_best_candidate_per_stream_count(result.all_candidates);
  result.decision = compare_sm_allocation_candidates(
      result.best_per_stream_count
  );

  return result;
}
