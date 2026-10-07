#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "00_pipeline_configuration.hpp"
#include "01_task_graph_analysis.hpp"
#include "02_task_importance.hpp"
#include "03_stream_scheduling.hpp"
#include "04_execution_configuration_selection.hpp"
#include "05_green_context_execution.cuh"
#include "bench_timer.hpp"

namespace {

std::filesystem::path stream_plot_directory() {
  const char* configured = std::getenv("TASK_GRAPH_STREAM_PLOT_DIR");
  return configured == nullptr
      ? std::filesystem::path("stream_plots")
      : std::filesystem::path(configured);
}

std::string format_sm_counts(const std::vector<int>& sm_counts) {
  std::ostringstream output;
  output << '[';

  for (std::size_t i = 0; i < sm_counts.size(); ++i) {
    if (i != 0) {
      output << ',';
    }
    output << sm_counts[i];
  }

  output << ']';
  return output.str();
}

void print_stg_summary(const std::vector<TaskSpec>& tasks) {
  stg::print_stg_summary(tasks);
  std::cout
      << "task_grid     : " << kTaskParallelSmLimit << " blocks\n"
      << "task_SM_limit : " << kTaskParallelSmLimit << " SM\n";
}

void print_task_importance(
    const std::vector<TaskSpec>& tasks,
    const TaskImportanceResult& importance
) {
  const std::streamsize previous_precision =
      std::cout.precision();

  std::cout << std::setprecision(
      std::numeric_limits<double>::max_digits10
  );

  std::cout << "\n===== Task importance =====\n";

  for (const auto& task : tasks) {
    const auto importance_it =
        importance.bottom_levels.find(task.id);

    if (importance_it == importance.bottom_levels.end()) {
      throw std::runtime_error(
          "bottom level not found for task: " +
          std::to_string(task.id)
      );
    }

    std::cout
        << "task " << task.id
        << " : importance=" << importance_it->second
        << '\n';
  }

  std::cout << "===========================\n";
  std::cout.precision(previous_precision);
}

void print_stream_candidate_comparison(
    const std::vector<SmAllocationCandidate>& all_candidates,
    const std::vector<SmAllocationCandidate>& best_per_stream_count,
    const SmAllocationDecision& decision
) {
  std::cout
      << "\n===== All SM allocation candidates =====\n"
      << "smaller estimated makespan is better\n";

  for (const auto& candidate : all_candidates) {
    bool best_for_stream_count = false;

    for (const auto& best : best_per_stream_count) {
      if (best.sm_counts == candidate.sm_counts &&
          std::abs(
              best.estimated_makespan - candidate.estimated_makespan
          ) <= 1.0e-9) {
        best_for_stream_count = true;
        break;
      }
    }

    std::cout
        << "Stream=" << candidate.sm_counts.size()
        << " SM=" << std::setw(16)
        << format_sm_counts(candidate.sm_counts)
        << " estimated_makespan="
        << candidate.estimated_makespan;

    if (best_for_stream_count) {
      std::cout << "  <-- BEST FOR "
                << candidate.sm_counts.size()
                << " STREAM";
    }

    std::cout << '\n';
  }

  std::cout << "\n===== Best result for each Stream count =====\n";

  for (const auto& candidate : best_per_stream_count) {
    const bool selected = candidate.sm_counts == decision.sm_counts;

    std::cout
        << "Stream=" << candidate.sm_counts.size()
        << " SM=" << std::setw(16)
        << format_sm_counts(candidate.sm_counts)
        << " estimated_makespan="
        << candidate.estimated_makespan;

    if (selected) {
      std::cout << "  <-- SELECTED";
    }

    std::cout << '\n';
  }

  std::cout << "=============================================\n\n";
}

void write_stream_makespan_csv(
    const std::string& stg_path,
    const std::vector<SmAllocationCandidate>& all_candidates,
    const SmAllocationDecision& decision
) {
  namespace fs = std::filesystem;

  fs::create_directories(stream_plot_directory());

  const std::string stem = fs::path(stg_path).stem().string();
  const fs::path csv_path =
      stream_plot_directory() /
      (stem + "_stream_makespan.csv");

  std::ofstream output(csv_path);

  if (!output) {
    throw std::runtime_error(
        "failed to open stream comparison CSV: " +
        csv_path.string()
    );
  }

  output << "stream_count,sm_counts,estimated_makespan,selected\n";
  output << std::setprecision(15);

  for (const auto& candidate : all_candidates) {
    const bool selected = candidate.sm_counts == decision.sm_counts;

    output
        << candidate.sm_counts.size() << ','
        << '"' << format_sm_counts(candidate.sm_counts) << '"' << ','
        << candidate.estimated_makespan << ','
        << (selected ? 1 : 0) << '\n';
  }

  std::cout << "stream comparison CSV: "
            << csv_path.string() << '\n';
}

void generate_stream_makespan_plot(const std::string& stg_path) {
  if (std::getenv("STG_DISABLE_STREAM_PLOT") != nullptr) {
    return;
  }

  namespace fs = std::filesystem;

  const std::string stem = fs::path(stg_path).stem().string();
  const fs::path csv_path =
      stream_plot_directory() /
      (stem + "_stream_makespan.csv");
  const fs::path png_path =
      stream_plot_directory() /
      (stem + "_stream_makespan.png");

  std::ostringstream command;
  command
      << "python3 plot_stream_makespan.py "
      << '"' << csv_path.string() << '"' << ' '
      << '"' << png_path.string() << '"';

  const int status = std::system(command.str().c_str());

  if (status == 0) {
    std::cout << "stream comparison graph: "
              << png_path.string() << '\n';
  }
  else {
    std::cerr
        << "warning: failed to generate stream comparison graph.\n"
        << "run manually: " << command.str() << '\n';
  }
}

void print_configuration(
    const std::vector<TaskSpec>& tasks,
    const SmAllocationDecision& decision,
    const PipelineOptions& options
) {
  const int stream_count =
      static_cast<int>(decision.sm_counts.size());

  double sequential_estimated_time = 0.0;

  for (const auto& task : tasks) {
    sequential_estimated_time +=
        static_cast<double>(std::max(0, task.proc_time));
  }

  const double estimated_speedup =
      decision.estimated_makespan <= 0.0
          ? 0.0
          : sequential_estimated_time /
                decision.estimated_makespan;

  const double estimated_reduction_percent =
      sequential_estimated_time <= 0.0
          ? 0.0
          : (sequential_estimated_time -
             decision.estimated_makespan) /
                sequential_estimated_time * 100.0;

  std::cout
      << "===== Proposed configuration =====\n"
      << "max_stream_count   : " << options.max_stream_count << '\n'
      << "stream_count       : " << stream_count << '\n'
      << "selected SM        : "
      << format_sm_counts(decision.sm_counts) << '\n'
      << "estimated makespan : "
      << decision.estimated_makespan << '\n'
      << "sequential estimate: "
      << sequential_estimated_time << '\n'
      << "estimated speedup  : "
      << estimated_speedup << " x\n"
      << "estimated reduction: "
      << estimated_reduction_percent << " %\n";
}

struct StreamProcessingTimeSummary {
  int stream_id = 0;
  int sm_count = 0;
  int allocation_model_sm_count = 0;
  int task_count = 0;
  double proc_time_before_sm_partition = 0.0;
  double relative_load_after_sm_partition = 0.0;
};

std::vector<StreamProcessingTimeSummary>
summarize_stream_processing_times(
    const std::vector<TaskSpec>& tasks,
    const SmAllocationCandidate& candidate
) {
  const int stream_count =
      static_cast<int>(candidate.sm_counts.size());

  std::vector<StreamProcessingTimeSummary> summaries;
  summaries.reserve(static_cast<std::size_t>(stream_count));

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    const int runtime_sm_count =
        candidate.sm_counts.at(static_cast<std::size_t>(stream_id));
    const int allocation_model_sm_count =
        runtime_sm_count;

    summaries.push_back(StreamProcessingTimeSummary{
        stream_id,
        runtime_sm_count,
        allocation_model_sm_count,
        0,
        0.0,
        0.0
    });
  }

  /*
   * タスク配置は変えず、64 SMで飽和する実測モデルを使って、
   * 最終SM配分における各Streamの処理時間を集計する。
   */
  for (const auto& task : tasks) {
    const int stream_id = candidate.schedule.task_stream.at(task.id);
    const std::size_t stream_index =
        static_cast<std::size_t>(stream_id);
    auto& summary = summaries.at(stream_index);
    const double proc_time =
        static_cast<double>(get_task_proc_time(task));

    ++summary.task_count;
    summary.proc_time_before_sm_partition += proc_time;
    summary.relative_load_after_sm_partition +=
        estimate_task_proc_time_on_stream(
            task,
            summary.allocation_model_sm_count,
            kSchedulingReferenceSmCount
        );
  }

  return summaries;
}

void print_processing_times_for_each_stream_count(
    const std::vector<TaskSpec>& tasks,
    const std::vector<SmAllocationCandidate>& candidates,
    const SmAllocationDecision& decision
) {
  std::cout
      << "===== Processing times for each stream count =====\n"
      << "task time = raw proc_time * measured block-pass multiplier\n"
      << "initial SM: 1=114, 2=56, 3=32, 4=24, 5=16\n";

  const std::streamsize previous_precision =
      std::cout.precision();
  std::cout << std::setprecision(
      std::numeric_limits<double>::max_digits10
  );

  for (const auto& candidate : candidates) {
    const bool selected = candidate.sm_counts == decision.sm_counts;
    const auto summaries = summarize_stream_processing_times(
        tasks,
        candidate
    );

    std::cout
        << "Stream count=" << candidate.sm_counts.size()
        << " SM=" << format_sm_counts(candidate.sm_counts);
    if (selected) {
      std::cout << "  <-- SELECTED";
    }
    std::cout << '\n';

    if (candidate.sm_counts.size() >= 2) {
      const auto stages = optimize_stream_sm_allocation(
          tasks,
          candidate.schedule
      );
      std::cout
          << "  initial-pool optimum : "
          << format_sm_counts(
              stages.initial_pool_optimized_sm_counts
          ) << '\n'
          << "  after stream0 +2     : "
          << format_sm_counts(
              stages.after_stream_0_remainder_sm_counts
          ) << '\n'
          << "  final with remainder : "
          << format_sm_counts(stages.runtime_sm_counts)
          << '\n';
    }

    for (const auto& summary : summaries) {
      std::cout
          << "  stream " << summary.stream_id
          << " : final_SM=" << summary.sm_count
          << ", tasks=" << summary.task_count
          << ", proc_time_before_sm_partition="
          << summary.proc_time_before_sm_partition
          << ", relative_load_after_sm_partition="
          << summary.relative_load_after_sm_partition
          << '\n';
    }
  }

  std::cout.precision(previous_precision);
  std::cout << "==================================================\n";
}

void write_processing_times_for_each_stream_count_csv(
    const std::string& stg_path,
    const std::vector<TaskSpec>& tasks,
    const std::vector<SmAllocationCandidate>& candidates,
    const SmAllocationDecision& decision
) {
  namespace fs = std::filesystem;

  fs::create_directories(stream_plot_directory());

  const std::string stem = fs::path(stg_path).stem().string();
  const fs::path csv_path =
      stream_plot_directory() /
      (stem + "_stream_count_processing_times.csv");

  std::ofstream output(csv_path);

  if (!output) {
    throw std::runtime_error(
        "failed to open stream-count processing-time CSV: " +
        csv_path.string()
    );
  }

  output
      << "stream_count,stream_id,initial_pool_sm,"
      << "after_stream0_plus_2_sm,sm_count,task_count,"
      << "allocation_model_sm_count,"
      << "raw_proc_time_sum,"
      << "before_redistribution_processing_time,"
      << "first_stage_processing_time,"
      << "relative_load_after_sm_partition,"
      << "before_redistribution_makespan,"
      << "first_stage_makespan,final_makespan,selected\n";
  output << std::setprecision(15);

  for (const auto& candidate : candidates) {
    const bool selected = candidate.sm_counts == decision.sm_counts;
    const auto summaries = summarize_stream_processing_times(
        tasks,
        candidate
    );
    std::vector<int> initial_pool_sm_counts = candidate.sm_counts;
    std::vector<int> after_stream_0_plus_2_sm_counts =
        candidate.sm_counts;
    const int stream_count =
        static_cast<int>(candidate.sm_counts.size());
    const std::vector<int> initial_equal_sm_counts(
        candidate.sm_counts.size(),
        initial_sm_per_stream_for_count(stream_count)
    );
    const double before_redistribution_makespan =
        estimate_stream_load_makespan(
            tasks,
            candidate.schedule,
            initial_equal_sm_counts
        );
    double first_stage_makespan = candidate.estimated_makespan;

    if (candidate.sm_counts.size() >= 2) {
      const auto stages = optimize_stream_sm_allocation(
          tasks,
          candidate.schedule
      );
      initial_pool_sm_counts =
          stages.initial_pool_optimized_sm_counts;
      after_stream_0_plus_2_sm_counts =
          stages.after_stream_0_remainder_sm_counts;
      first_stage_makespan = stages.initial_pool_makespan;
    }

    for (const auto& summary : summaries) {
      const std::size_t stream_index =
          static_cast<std::size_t>(summary.stream_id);
      const double before_redistribution_processing_time =
          estimate_stream_processing_time_sum(
              tasks,
              candidate.schedule,
              initial_equal_sm_counts,
              summary.stream_id
          );
      const double first_stage_processing_time =
          estimate_stream_processing_time_sum(
              tasks,
              candidate.schedule,
              initial_pool_sm_counts,
              summary.stream_id
          );
      output
          << candidate.sm_counts.size() << ','
          << summary.stream_id << ','
          << initial_pool_sm_counts.at(stream_index) << ','
          << after_stream_0_plus_2_sm_counts.at(stream_index) << ','
          << summary.sm_count << ','
          << summary.task_count << ','
          << summary.allocation_model_sm_count << ','
          << summary.proc_time_before_sm_partition << ','
          << before_redistribution_processing_time << ','
          << first_stage_processing_time << ','
          << summary.relative_load_after_sm_partition << ','
          << before_redistribution_makespan << ','
          << first_stage_makespan << ','
          << candidate.estimated_makespan << ','
          << (selected ? 1 : 0) << '\n';
    }
  }

  std::cout << "stream-count processing-time CSV: "
            << csv_path.string() << '\n';
}

BenchResult run_pipeline(const std::string& stg_path) {
  BenchResult benchmark;
  const auto total_start = Clock::now();
  const PipelineOptions options =
      read_pipeline_options_from_environment();

  /* Stage 1: STG解析 */
  StgAnalysisResult analysis;
  {
    ScopedTimer timer(benchmark.stg_analysis_ms);
    analysis = analyze_stg(stg_path);
  }

  print_stg_summary(analysis.tasks);

  /* Stage 2: proc_timeと後続タスクから重要度を1回だけ計算 */
  TaskImportanceResult importance;
  {
    ScopedTimer timer(benchmark.task_importance_ms);
    importance = evaluate_task_importance(
        analysis.tasks
    );
  }

  print_task_importance(
      analysis.tasks,
      importance
  );

  /*
   * SM配分候補は00_pipeline_configuration.hppで定義済み。
   * 1～5 Streamを常に比較する。
   * STG_MAX_STREAMSが指定された場合だけ、その値を上限とする。
   */
  const int stream_limit = options.max_stream_count;

  const std::vector<std::vector<int>> sm_count_candidates =
      make_sm_count_candidates(
          stream_limit,
          options
      );

  if (sm_count_candidates.empty()) {
    throw std::runtime_error("no SM allocation candidates");
  }

  /* Stage 3: SM数を使わず、Stream数ごとにタスクを配置する。 */
  std::vector<StreamScheduleResult> placements;
  placements.reserve(sm_count_candidates.size());

  {
    ScopedTimer timer(benchmark.stream_placement_ms);

    for (const auto& sm_counts : sm_count_candidates) {
      const int stream_count =
          static_cast<int>(sm_counts.size());

      const auto existing = std::find_if(
          placements.begin(),
          placements.end(),
          [stream_count](const StreamScheduleResult& placement) {
            return placement.stream_count == stream_count;
          }
      );

      if (existing != placements.end()) {
        continue;
      }

      placements.push_back(
          place_tasks_on_streams(
              analysis.tasks,
              importance,
              stream_count
          )
      );
    }
  }

  /* Stage 4: Stage 3の配置を固定し、SM配分を決定する。 */
  const bool optimize_default_stream_allocation =
      !options.disable_gc &&
      !options.stream_sm_counts.has_value() &&
      !options.two_stream_gc_sm.has_value();

  SmAllocationStageResult allocation;

  {
    ScopedTimer timer(benchmark.sm_allocation_ms);
    allocation = allocate_sm_resources(
        analysis.tasks,
        placements,
        sm_count_candidates,
        optimize_default_stream_allocation
    );
  }

  const auto& all_candidates = allocation.all_candidates;
  const auto& best_per_stream_count =
      allocation.best_per_stream_count;
  const auto& decision = allocation.decision;

  /* Stage 4が決めたSM配分を実機で構成できるか確認する。 */
  if (!options.disable_gc) {
    const SmPartitionInfo partition_info = query_sm_partition_info();
    validate_fixed_sm_table_compatibility(partition_info);

    for (const auto& candidate : all_candidates) {
      validate_green_context_sm_counts(
          candidate.sm_counts,
          partition_info
      );
    }
  }

  print_stream_candidate_comparison(
      all_candidates,
      best_per_stream_count,
      decision
  );

  print_configuration(
      analysis.tasks,
      decision,
      options
  );

  /* run_benchmark_suite.shから要求された場合だけ追加集計を出力する。 */
  if (std::getenv("STG_PRINT_STREAM_PROCESSING_TIMES") != nullptr) {
    print_processing_times_for_each_stream_count(
        analysis.tasks,
        best_per_stream_count,
        decision
    );
    write_processing_times_for_each_stream_count_csv(
        stg_path,
        analysis.tasks,
        best_per_stream_count,
        decision
    );
  }

  /* Stage 5: 選択した構成を変更せずGPU上で実行 */
  GreenContextExecutionResult execution;

  {
    ScopedTimer timer(benchmark.green_context_execution_ms);

    execution = execute_with_green_context(
        analysis.tasks,
        decision,
        GreenContextExecutionOptions{
            options.disable_gc,
            options.background_chunk_count,
            kSchedulingReferenceSmCount
        }
    );
  }

  benchmark.gpu_submit_wait_ms = execution.gpu_submit_wait_ms;
  benchmark.gpu_kernel_ms = execution.gpu_kernel_ms;
  benchmark.total_ms = elapsed_ms(total_start, Clock::now());

  write_stream_makespan_csv(
      stg_path,
      all_candidates,
      decision
  );
  generate_stream_makespan_plot(stg_path);

  return benchmark;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " input.stg\n";
    return 1;
  }

  try {
    const BenchResult result = run_pipeline(argv[1]);
    std::cout << "mode: proposed\n";
    print_result(result);
  }
  catch (const std::exception& error) {
    std::cerr << "exception: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
