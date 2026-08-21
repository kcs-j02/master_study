#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "01_stg_analysis.hpp"
#include "02_task_importance.hpp"
#include "03_stream_assignment.hpp"

/*
 * Stage 4の予測で使用するGH100向け固定SM総数。
 */
inline constexpr int kSchedulingReferenceSmCount = 114;
inline constexpr int kMaximumConfiguredStreamCount = 5;

/*
 * Stage 3で評価済みのSM配分候補。
 */
struct SmAllocationCandidate {
  std::vector<int> sm_counts;
  StreamScheduleResult schedule;
};

/*
 * Stage 4が選択したSM配分と、その予測結果。
 */
struct SmAllocationDecision {
  std::vector<int> sm_counts;
  StreamScheduleResult schedule;
  double estimated_makespan = std::numeric_limits<double>::max();
};

/*
 * 実行パイプラインの環境変数設定。
 * background_chunk_countはStage 5の実行設定だが、構成を一か所で
 * 読み取れるようStage 4の結果として保持する。
 */
struct PipelineOptions {
  int max_stream_count = kMaximumConfiguredStreamCount;
  bool disable_gc = false;
  int background_chunk_count = 1;
  std::optional<int> two_stream_gc_sm;
  std::optional<std::vector<int>> stream_sm_counts;
};

/*
 * Green Contextで利用可能なSM資源と分割粒度。
 */
struct SmPartitionInfo {
  int available_sm = 0;
  int unit_sm = 0;
  int min_group_sm = 0;
};

inline int align_up_int(int value, int alignment) {
  if (alignment <= 0) {
    throw std::invalid_argument("alignment must be positive");
  }

  return ((value + alignment - 1) / alignment) * alignment;
}

inline int align_down_int(int value, int alignment) {
  if (alignment <= 0) {
    throw std::invalid_argument("alignment must be positive");
  }

  return (value / alignment) * alignment;
}

namespace sm_allocation_detail {

inline void check_cuda(
    cudaError_t error,
    const char* operation
) {
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " +
        cudaGetErrorString(error)
    );
  }
}

inline int parse_integer(
    const char* value,
    const char* variable_name
) {
  try {
    return std::stoi(value);
  } catch (const std::invalid_argument&) {
    throw std::runtime_error(
        std::string(variable_name) + " must be an integer"
    );
  } catch (const std::out_of_range&) {
    throw std::runtime_error(
        std::string(variable_name) + " is out of range"
    );
  }
}

inline std::vector<int> parse_sm_count_list(const char* value) {
  std::vector<int> counts;
  std::istringstream input(value);
  std::string token;

  while (std::getline(input, token, ',')) {
    try {
      counts.push_back(std::stoi(token));
    } catch (const std::invalid_argument&) {
      throw std::runtime_error(
          "STG_STREAM_SM_COUNTS must contain integers"
      );
    } catch (const std::out_of_range&) {
      throw std::runtime_error(
          "STG_STREAM_SM_COUNTS contains an out-of-range value"
      );
    }
  }

  return counts;
}

}  // namespace sm_allocation_detail

/*
 * 実機からSM数、分割単位、Green Contextの最小SM数を取得する。
 * Stage 5もこの型と関数を再利用する。
 */
inline SmPartitionInfo query_sm_partition_info(
    int gpu_device_index = 0
) {
  sm_allocation_detail::check_cuda(
      cudaSetDevice(gpu_device_index),
      "cudaSetDevice failed"
  );

  cudaDevResource initial_sm_resource{};

  sm_allocation_detail::check_cuda(
      cudaDeviceGetDevResource(
          gpu_device_index,
          &initial_sm_resource,
          cudaDevResourceTypeSm
      ),
      "cudaDeviceGetDevResource failed"
  );

  const int available_sm =
      static_cast<int>(initial_sm_resource.sm.smCount);
  const int min_partition_size =
      static_cast<int>(initial_sm_resource.sm.minSmPartitionSize);
  const int coscheduled_alignment =
      static_cast<int>(initial_sm_resource.sm.smCoscheduledAlignment);
  const int unit_sm = std::max(2, coscheduled_alignment);
  const int min_group_sm = align_up_int(
      std::max(2, min_partition_size),
      unit_sm
  );

  if (available_sm <= 0 || unit_sm <= 0 || min_group_sm <= 0) {
    throw std::runtime_error("invalid SM partition information");
  }

  return {available_sm, unit_sm, min_group_sm};
}

/*
 * 現在の固定SM表を使用できる114-SM GPUか検証する。
 */
inline void validate_fixed_sm_table_compatibility(
    const SmPartitionInfo& partition_info
) {
  if (partition_info.available_sm != kSchedulingReferenceSmCount) {
    throw std::runtime_error(
        "initial SM table requires a 114-SM GPU"
    );
  }
}

/*
 * Stage 5へ渡す前に、Green Contextとして実現可能なSM配分か確認する。
 * Stream 0はprimary context、Stream 1以降はGreen Contextである。
 */
inline void validate_green_context_sm_counts(
    const std::vector<int>& stream_sm_counts,
    const SmPartitionInfo& partition_info
) {
  validate_stream_sm_counts(stream_sm_counts);

  if (stream_sm_counts.size() == 1) {
    if (stream_sm_counts.front() != partition_info.available_sm) {
      throw std::invalid_argument(
          "single normal stream must use all available SMs"
      );
    }
    return;
  }

  int green_context_sm_total = 0;

  for (std::size_t stream_id = 1;
       stream_id < stream_sm_counts.size();
       ++stream_id) {
    const int sm_count = stream_sm_counts.at(stream_id);

    if (sm_count < partition_info.min_group_sm) {
      throw std::invalid_argument(
          "GC SM count is smaller than minimum: stream " +
          std::to_string(stream_id)
      );
    }
    if (sm_count % partition_info.unit_sm != 0) {
      throw std::invalid_argument(
          "GC SM count is not aligned: stream " +
          std::to_string(stream_id)
      );
    }

    green_context_sm_total += sm_count;
  }

  if (green_context_sm_total >= partition_info.available_sm) {
    throw std::invalid_argument(
        "Green Contexts must leave SMs for the normal stream"
    );
  }
}

/*
 * 最大レベル幅を取得する。
 */
inline int get_max_level_width(const TaskLevels& levels) {
  int max_width = 0;

  for (const auto& level : levels) {
    max_width = std::max(
        max_width,
        static_cast<int>(level.size())
    );
  }

  if (max_width <= 0) {
    throw std::runtime_error("task levels are empty");
  }

  return max_width;
}

/*
 * レベル幅と設定上限から使用するStream数を決める。
 */
inline int decide_stream_count(
    const TaskLevels& levels,
    int max_stream_count
) {
  if (max_stream_count <= 0 ||
      max_stream_count > kMaximumConfiguredStreamCount) {
    throw std::invalid_argument(
        "max_stream_count must be between 1 and 5"
    );
  }

  return std::min(get_max_level_width(levels), max_stream_count);
}

inline int decide_stream_count(
    const TaskLevels& levels,
    const PipelineOptions& options
) {
  return decide_stream_count(levels, options.max_stream_count);
}

/*
 * GH100（114 SM）向けの固定SM配分。
 * Stream 0をprimary context、Stream 1以降をGreen Contextとして扱う。
 */
inline std::vector<int> make_initial_stream_sm_counts(
    int stream_count
) {
  switch (stream_count) {
    case 1:
      return {114};
    case 2:
      return {82, 32};
    case 3:
      return {82, 16, 16};
    case 4:
      return {82, 16, 8, 8};
    case 5:
      return {82, 8, 8, 8, 8};
    default:
      throw std::invalid_argument(
          "stream_count must be between 1 and 5"
      );
  }
}

inline std::vector<int> make_initial_stream_sm_counts(
    int stream_count,
    const SmPartitionInfo& partition_info
) {
  validate_fixed_sm_table_compatibility(partition_info);
  return make_initial_stream_sm_counts(stream_count);
}

/*
 * Stream数ごとにStage 4で比較する固定SM配分候補を返す。
 * 3 Streamでは、均等寄りの82/16/16に加えて、
 * 優先度の高いGCへ多く割り当てる82/24/8も評価する。
 * Stream 1とStream 2の役割はStage 3の優先度順割当で決まるため、
 * 82/8/24は重複候補として追加しない。
 */
inline std::vector<std::vector<int>>
make_stream_sm_count_candidates(
    int stream_count
) {
  switch (stream_count) {
    case 1:
      return {{114}};
    case 2:
      return {{82, 32}};
    case 3:
      return {
          {82, 16, 16},
          {82, 24, 8}
      };
    case 4:
      return {{82, 16, 8, 8}};
    case 5:
      return {{82, 8, 8, 8, 8}};
    default:
      throw std::invalid_argument(
          "stream_count must be between 1 and 5"
      );
  }
}

/*
 * 1 Streamから指定上限までの固定SM配分候補を入力順に作る。
 * 3 Streamの場合だけ複数のSM配分候補を追加する。
 */
inline std::vector<std::vector<int>> make_fixed_sm_count_candidates(
    const TaskLevels& levels,
    int max_stream_count
) {
  const int candidate_limit =
      decide_stream_count(levels, max_stream_count);
  std::vector<std::vector<int>> candidates;

  for (int stream_count = 1;
       stream_count <= candidate_limit;
       ++stream_count) {
    const auto stream_candidates =
        make_stream_sm_count_candidates(stream_count);

    candidates.insert(
        candidates.end(),
        stream_candidates.begin(),
        stream_candidates.end()
    );
  }

  return candidates;
}

/*
 * 現行mainで使用している環境変数をまとめて解析する。
 */
inline PipelineOptions read_pipeline_options_from_environment() {
  PipelineOptions options;

  if (const char* value = std::getenv("STG_MAX_STREAMS")) {
    options.max_stream_count = sm_allocation_detail::parse_integer(
        value,
        "STG_MAX_STREAMS"
    );

    if (options.max_stream_count < 1 ||
        options.max_stream_count > kMaximumConfiguredStreamCount) {
      throw std::runtime_error(
          "STG_MAX_STREAMS must be between 1 and 5"
      );
    }
  }

  options.disable_gc = std::getenv("STG_DISABLE_GC") != nullptr;

  if (const char* value = std::getenv("STG_BACKGROUND_CHUNKS")) {
    options.background_chunk_count =
        sm_allocation_detail::parse_integer(
            value,
            "STG_BACKGROUND_CHUNKS"
        );

    if (options.background_chunk_count < 1 ||
        options.background_chunk_count > 16) {
      throw std::runtime_error(
          "STG_BACKGROUND_CHUNKS must be between 1 and 16"
      );
    }
  }

  if (const char* value = std::getenv("STG_TWO_STREAM_GC_SM")) {
    const int requested_gc_sm = sm_allocation_detail::parse_integer(
        value,
        "STG_TWO_STREAM_GC_SM"
    );

    if (requested_gc_sm <= 0 ||
        requested_gc_sm >= kSchedulingReferenceSmCount) {
      throw std::runtime_error(
          "STG_TWO_STREAM_GC_SM must leave SMs for normal stream"
      );
    }

    options.two_stream_gc_sm = requested_gc_sm;
  }

  if (const char* value = std::getenv("STG_STREAM_SM_COUNTS")) {
    std::vector<int> requested_counts =
        sm_allocation_detail::parse_sm_count_list(value);
    const long long requested_total = std::accumulate(
        requested_counts.begin(),
        requested_counts.end(),
        0LL
    );

    if (requested_total > kSchedulingReferenceSmCount ||
        std::any_of(
            requested_counts.begin(),
            requested_counts.end(),
            [](int count) { return count <= 0; }
        )) {
      throw std::runtime_error(
          "STG_STREAM_SM_COUNTS must be positive and total at most 114"
      );
    }

    options.stream_sm_counts = std::move(requested_counts);
  }

  return options;
}

/*
 * 固定表へ環境変数の上書きを現行mainと同じ順序で適用する。
 * STG_STREAM_SM_COUNTSが最も強く、次にSTG_TWO_STREAM_GC_SM、
 * STG_DISABLE_GCの順となる。
 */
inline std::vector<int> resolve_stream_sm_counts(
    int stream_count,
    const PipelineOptions& options,
    int reference_sm_count = kSchedulingReferenceSmCount
) {
  if (reference_sm_count != kSchedulingReferenceSmCount) {
    throw std::invalid_argument(
        "fixed SM table requires reference_sm_count=114"
    );
  }

  std::vector<int> sm_counts =
      make_initial_stream_sm_counts(stream_count);

  if (options.disable_gc) {
    sm_counts.assign(
        static_cast<std::size_t>(stream_count),
        reference_sm_count
    );
  }

  if (options.two_stream_gc_sm.has_value()) {
    if (stream_count != 2) {
      throw std::runtime_error(
          "STG_TWO_STREAM_GC_SM requires a two-stream configuration"
      );
    }

    const int requested_gc_sm = *options.two_stream_gc_sm;

    if (requested_gc_sm <= 0 ||
        requested_gc_sm >= reference_sm_count) {
      throw std::runtime_error(
          "STG_TWO_STREAM_GC_SM must leave SMs for normal stream"
      );
    }

    sm_counts = {
        reference_sm_count - requested_gc_sm,
        requested_gc_sm
    };
  }

  if (options.stream_sm_counts.has_value()) {
    const auto& requested_counts = *options.stream_sm_counts;

    if (static_cast<int>(requested_counts.size()) != stream_count) {
      throw std::runtime_error(
          "STG_STREAM_SM_COUNTS size must match stream_count"
      );
    }

    validate_stream_sm_counts(requested_counts);

    const long long requested_total = std::accumulate(
        requested_counts.begin(),
        requested_counts.end(),
        0LL
    );

    if (requested_total > reference_sm_count) {
      throw std::runtime_error(
          "STG_STREAM_SM_COUNTS total must be at most 114"
      );
    }

    sm_counts = requested_counts;
  }

  return sm_counts;
}

/*
 * 1つのSM配分をStage 3で評価し、比較可能な候補にする。
 */
inline SmAllocationCandidate evaluate_sm_allocation_candidate(
    const std::vector<TaskSpec>& tasks,
    const TaskImportanceResult& importance,
    std::vector<int> sm_counts,
    int reference_sm_count = kSchedulingReferenceSmCount
) {
  StreamScheduleResult schedule = place_tasks_on_streams(
      tasks,
      importance,
      sm_counts,
      reference_sm_count
  );

  return {
      std::move(sm_counts),
      std::move(schedule)
  };
}

/*
 * SM配分群を入力順にStage 3で評価する。
 */
inline std::vector<SmAllocationCandidate>
evaluate_sm_allocation_candidates(
    const std::vector<TaskSpec>& tasks,
    const TaskImportanceResult& importance,
    const std::vector<std::vector<int>>& sm_count_candidates,
    int reference_sm_count = kSchedulingReferenceSmCount
) {
  if (sm_count_candidates.empty()) {
    throw std::invalid_argument(
        "sm_count_candidates must not be empty"
    );
  }

  std::vector<SmAllocationCandidate> candidates;
  candidates.reserve(sm_count_candidates.size());

  for (const auto& sm_counts : sm_count_candidates) {
    candidates.push_back(
        evaluate_sm_allocation_candidate(
            tasks,
            importance,
            sm_counts,
            reference_sm_count
        )
    );
  }

  return candidates;
}



/*
 * 同じStream数に複数のSM配分候補がある場合、その中で
 * 予測makespanが最小の候補だけを残す。
 * 例: 3 Streamでは82/16/16と82/24/8を比較し、速い方を残す。
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

      validate_stream_sm_counts(candidate.sm_counts);

      if (!std::isfinite(candidate.schedule.makespan) ||
          candidate.schedule.makespan < 0.0) {
        throw std::invalid_argument(
            "candidate makespan must be finite and non-negative"
        );
      }

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
 * Stage 3で評価済みの候補から、予測makespanが最小のものを選ぶ。
 * 同値ならStream数が少ない候補を選び、それも同じなら入力順を保つ。
 */
inline SmAllocationDecision compare_sm_allocation_candidates(
    const std::vector<SmAllocationCandidate>& candidates
) {
  if (candidates.empty()) {
    throw std::invalid_argument("candidates must not be empty");
  }

  constexpr double epsilon = 1.0e-9;
  std::size_t best_index = 0;

  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const auto& candidate = candidates.at(index);

    validate_stream_sm_counts(candidate.sm_counts);

    if (!std::isfinite(candidate.schedule.makespan) ||
        candidate.schedule.makespan < 0.0) {
      throw std::invalid_argument(
          "candidate makespan must be finite and non-negative"
      );
    }

    if (index == 0) {
      continue;
    }

    const auto& best = candidates.at(best_index);
    const double makespan_difference =
        candidate.schedule.makespan - best.schedule.makespan;

    if (makespan_difference < -epsilon ||
        (std::abs(makespan_difference) <= epsilon &&
         candidate.sm_counts.size() < best.sm_counts.size())) {
      best_index = index;
    }
  }

  const auto& best = candidates.at(best_index);

  return {
      best.sm_counts,
      best.schedule,
      best.schedule.makespan
  };
}

/*
 * SM配分群の評価と比較をまとめて行う便宜API。
 */
inline SmAllocationDecision compare_sm_allocations(
    const std::vector<TaskSpec>& tasks,
    const TaskImportanceResult& importance,
    const std::vector<std::vector<int>>& sm_count_candidates,
    int reference_sm_count = kSchedulingReferenceSmCount
) {
  return compare_sm_allocation_candidates(
      evaluate_sm_allocation_candidates(
          tasks,
          importance,
          sm_count_candidates,
          reference_sm_count
      )
  );
}
