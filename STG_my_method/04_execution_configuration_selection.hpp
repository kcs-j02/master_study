#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "00_pipeline_configuration.hpp"
#include "02_task_importance.hpp"
#include "03_stream_assignment.hpp"

/*
 * Stage 4: 予測makespanが最小の実行構成を選択する。
 *
 * SM配分候補そのものは00_pipeline_configuration.hppで定義する。
 * Stage 4は、Stage 2・3で評価済みの候補を比較するだけにする。
 */
struct SmAllocationCandidate {
  std::vector<int> sm_counts;
  TaskImportanceResult importance;
  StreamScheduleResult schedule;
};

struct SmAllocationDecision {
  std::vector<int> sm_counts;
  TaskImportanceResult importance;
  StreamScheduleResult schedule;
  double estimated_makespan = std::numeric_limits<double>::max();
};

inline void validate_sm_allocation_candidate(
    const SmAllocationCandidate& candidate
) {
  validate_stream_sm_counts(candidate.sm_counts);

  if (!std::isfinite(candidate.schedule.makespan) ||
      candidate.schedule.makespan < 0.0) {
    throw std::invalid_argument(
        "candidate makespan must be finite and non-negative"
    );
  }

  if (candidate.importance.stream_sm_counts != candidate.sm_counts) {
    throw std::invalid_argument(
        "candidate importance and SM allocation do not match"
    );
  }
}

/*
 * 同じStream数に複数候補がある場合、その中で最小makespanを残す。
 * 例: {82,16,16} と {82,24,8}
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

  for (int stream_count = 1;
       stream_count <= kMaximumConfiguredStreamCount;
       ++stream_count) {
    const SmAllocationCandidate* best = nullptr;

    for (const auto& candidate : candidates) {
      if (static_cast<int>(candidate.sm_counts.size()) != stream_count) {
        continue;
      }

      validate_sm_allocation_candidate(candidate);

      if (best == nullptr ||
          candidate.schedule.makespan <
              best->schedule.makespan - epsilon) {
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
        candidate.schedule.makespan - best.schedule.makespan;

    if (difference < -epsilon ||
        (std::abs(difference) <= epsilon &&
         candidate.sm_counts.size() < best.sm_counts.size())) {
      best_index = index;
    }
  }

  const auto& best = candidates[best_index];

  return {
      best.sm_counts,
      best.importance,
      best.schedule,
      best.schedule.makespan
  };
}
