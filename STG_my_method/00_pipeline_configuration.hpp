#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * 提案手法の各Stageで共通して使用するパイプライン設定。
 *
 * 共通定数、実行オプション、環境変数の読込み、および
 * Stage 2・3で評価するSM配分候補をここで定義する。
 */

/*
 * GPU全体のSM数。
 * SM配分候補の合計上限として使用する。
 */
inline constexpr int kSchedulingReferenceSmCount = 114;

/*
 * 同時に比較する最大Stream数。
 */
inline constexpr int kMaximumConfiguredStreamCount = 5;

/*
 * 1タスクに許可する最大SM数。
 *
 * GC解放後に最大114 SMまで利用できるようにする。
 */
inline constexpr int kTaskParallelSmLimit = 114;

/*
 * proc_timeを取得した基準SM数。
 *
 * proc_timeは64 SM相当を基準とする。
 *
 * Stage 2・3の予測では、
 * 現在のkernelについて64 SMを超えた領域で
 * 理想的な速度向上を仮定しない。
 */
inline constexpr int kProcTimeReferenceSmCount = 64;

/*
 * タスクの問題サイズ。
 *
 * kTaskParallelSmLimitを114へ変更しても、
 * タスクそのものの仕事量は従来と同じ
 *
 *   64 * 256
 *
 * のまま固定する。
 */
inline constexpr int kTaskElementCount = 64 * 256;


struct PipelineOptions {
  int max_stream_count =
      kMaximumConfiguredStreamCount;

  bool disable_gc = false;

  int background_chunk_count = 1;

  std::optional<int>
      two_stream_gc_sm;

  std::optional<std::vector<int>>
      stream_sm_counts;
};


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

  for (
      std::size_t stream_id = 0;
      stream_id < stream_sm_counts.size();
      ++stream_id
  ) {
    if (
        stream_sm_counts.at(stream_id) <= 0
    ) {
      throw std::invalid_argument(
          "stream SM count must be positive: stream " +
          std::to_string(stream_id)
      );
    }
  }
}


/*
 * ================================================================
 * あらかじめ定義するSM配分候補
 * ================================================================
 *
 * Stream 0:
 *   primary context
 *
 * Stream 1以降:
 *   Green Context
 */
inline std::vector<std::vector<int>>
make_stream_sm_count_candidates(
    int stream_count
) {
  switch (stream_count) {

    case 1:
      return {
          {114}
      };

    case 2:
      return {
          {82, 32}
      };

    case 3:
      return {
          {82, 16, 16},
          {82, 24, 8}
      };

    case 4:
      return {
          {82, 16, 8, 8}
      };

    case 5:
      return {
          {82, 8, 8, 8, 8}
      };

    default:
      throw std::invalid_argument(
          "stream_count must be between 1 and 5"
      );
  }
}


/*
 * 1 Streamからstream_limitまでの固定候補をまとめて返す。
 */
inline std::vector<std::vector<int>>
make_fixed_sm_count_candidates(
    int stream_limit
) {
  if (
      stream_limit <= 0 ||
      stream_limit >
          kMaximumConfiguredStreamCount
  ) {
    throw std::invalid_argument(
        "stream_limit must be between 1 and 5"
    );
  }


  std::vector<
      std::vector<int>
  > candidates;


  for (
      int stream_count = 1;
      stream_count <= stream_limit;
      ++stream_count
  ) {
    const auto stream_candidates =
        make_stream_sm_count_candidates(
            stream_count
        );


    candidates.insert(
        candidates.end(),
        stream_candidates.begin(),
        stream_candidates.end()
    );
  }


  return candidates;
}


namespace sm_config_detail {


inline int parse_integer(
    const char* value,
    const char* variable_name
) {
  try {
    return std::stoi(
        value
    );
  }

  catch (
      const std::invalid_argument&
  ) {
    throw std::runtime_error(
        std::string(variable_name) +
        " must be an integer"
    );
  }

  catch (
      const std::out_of_range&
  ) {
    throw std::runtime_error(
        std::string(variable_name) +
        " is out of range"
    );
  }
}


inline std::vector<int>
parse_sm_count_list(
    const char* value
) {
  std::vector<int> counts;

  std::istringstream input(
      value
  );

  std::string token;


  while (
      std::getline(
          input,
          token,
          ','
      )
  ) {
    try {
      counts.push_back(
          std::stoi(token)
      );
    }

    catch (
        const std::invalid_argument&
    ) {
      throw std::runtime_error(
          "STG_STREAM_SM_COUNTS must contain integers"
      );
    }

    catch (
        const std::out_of_range&
    ) {
      throw std::runtime_error(
          "STG_STREAM_SM_COUNTS contains an out-of-range value"
      );
    }
  }


  return counts;
}


}  // namespace sm_config_detail


/*
 * ================================================================
 * 環境変数を読み込む
 * ================================================================
 */
inline PipelineOptions
read_pipeline_options_from_environment() {
  PipelineOptions options;


  /*
   * 最大Stream数
   */
  if (
      const char* value =
          std::getenv(
              "STG_MAX_STREAMS"
          )
  ) {
    options.max_stream_count =
        sm_config_detail::
            parse_integer(
                value,
                "STG_MAX_STREAMS"
            );


    if (
        options.max_stream_count < 1 ||
        options.max_stream_count >
            kMaximumConfiguredStreamCount
    ) {
      throw std::runtime_error(
          "STG_MAX_STREAMS must be between 1 and 5"
      );
    }
  }


  /*
   * GC無効化
   */
  options.disable_gc =
      std::getenv(
          "STG_DISABLE_GC"
      ) != nullptr;


  /*
   * background kernel分割数
   */
  if (
      const char* value =
          std::getenv(
              "STG_BACKGROUND_CHUNKS"
          )
  ) {
    options.background_chunk_count =
        sm_config_detail::
            parse_integer(
                value,
                "STG_BACKGROUND_CHUNKS"
            );


    if (
        options.background_chunk_count < 1 ||
        options.background_chunk_count > 16
    ) {
      throw std::runtime_error(
          "STG_BACKGROUND_CHUNKS must be between 1 and 16"
      );
    }
  }


  /*
   * 2 Stream構成のGC SM数
   */
  if (
      const char* value =
          std::getenv(
              "STG_TWO_STREAM_GC_SM"
          )
  ) {
    const int gc_sm =
        sm_config_detail::
            parse_integer(
                value,
                "STG_TWO_STREAM_GC_SM"
            );


    if (
        gc_sm <= 0 ||
        gc_sm >=
            kSchedulingReferenceSmCount
    ) {
      throw std::runtime_error(
          "STG_TWO_STREAM_GC_SM must leave SMs for normal stream"
      );
    }


    options.two_stream_gc_sm =
        gc_sm;
  }


  /*
   * SM配分直接指定
   */
  if (
      const char* value =
          std::getenv(
              "STG_STREAM_SM_COUNTS"
          )
  ) {
    auto counts =
        sm_config_detail::
            parse_sm_count_list(
                value
            );


    validate_stream_sm_counts(
        counts
    );


    const long long total =
        std::accumulate(
            counts.begin(),
            counts.end(),
            0LL
        );


    if (
        total >
        kSchedulingReferenceSmCount
    ) {
      throw std::runtime_error(
          "STG_STREAM_SM_COUNTS total must be at most 114"
      );
    }


    options.stream_sm_counts =
        std::move(
            counts
        );
  }


  return options;
}


/*
 * ================================================================
 * 実際に評価するSM配分候補を作る
 * ================================================================
 *
 * 明示指定がある場合はその構成だけを評価する。
 */
inline std::vector<std::vector<int>>
make_sm_count_candidates(
    int stream_limit,
    const PipelineOptions& options
) {
  if (
      stream_limit <= 0 ||
      stream_limit >
          kMaximumConfiguredStreamCount
  ) {
    throw std::invalid_argument(
        "stream_limit must be between 1 and 5"
    );
  }


  /*
   * SM配分直接指定
   */
  if (
      options.stream_sm_counts.has_value()
  ) {
    const auto& counts =
        *options.stream_sm_counts;


    if (
        static_cast<int>(
            counts.size()
        ) >
        stream_limit
    ) {
      throw std::runtime_error(
          "STG_STREAM_SM_COUNTS exceeds the usable stream count"
      );
    }


    return {
        counts
    };
  }


  /*
   * 2 Stream SM指定
   */
  if (
      options.two_stream_gc_sm.has_value()
  ) {
    if (
        stream_limit < 2
    ) {
      throw std::runtime_error(
          "STG_TWO_STREAM_GC_SM requires at least two usable streams"
      );
    }


    const int gc_sm =
        *options.two_stream_gc_sm;


    return {
        {
            kSchedulingReferenceSmCount -
                gc_sm,

            gc_sm
        }
    };
  }


  /*
   * GCなし比較実験
   */
  if (
      options.disable_gc
  ) {
    std::vector<
        std::vector<int>
    > candidates;


    for (
        int stream_count = 1;
        stream_count <= stream_limit;
        ++stream_count
    ) {
      candidates.emplace_back(
          static_cast<std::size_t>(
              stream_count
          ),
          kSchedulingReferenceSmCount
      );
    }


    return candidates;
  }


  /*
   * 提案手法
   */
  return
      make_fixed_sm_count_candidates(
          stream_limit
      );
}