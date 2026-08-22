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
#include "01_stg_analysis.hpp"
#include "02_task_importance.hpp"
#include "03_stream_assignment.hpp"
#include "04_execution_configuration_selection.hpp"
#include "05_green_context_execution.cuh"
#include "bench_timer.hpp"

namespace {

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
              best.schedule.makespan - candidate.schedule.makespan
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
        << candidate.schedule.makespan;

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
        << candidate.schedule.makespan;

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

  fs::create_directories("stream_plots");

  const std::string stem = fs::path(stg_path).stem().string();
  const fs::path csv_path =
      fs::path("stream_plots") /
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
        << candidate.schedule.makespan << ','
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
      fs::path("stream_plots") /
      (stem + "_stream_makespan.csv");
  const fs::path png_path =
      fs::path("stream_plots") /
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

  std::vector<int> task_counts(
      static_cast<std::size_t>(stream_count),
      0
  );
  std::vector<long long> proc_loads(
      static_cast<std::size_t>(stream_count),
      0
  );

  for (const auto& task : tasks) {
    const int stream_id = decision.schedule.task_stream.at(task.id);
    ++task_counts.at(static_cast<std::size_t>(stream_id));
    proc_loads.at(static_cast<std::size_t>(stream_id)) +=
        std::max(0, task.proc_time);
  }

  std::cout << "===== Planned stream load =====\n";

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    std::cout
        << "stream " << stream_id
        << " : SM="
        << decision.sm_counts.at(static_cast<std::size_t>(stream_id))
        << ", tasks="
        << task_counts.at(static_cast<std::size_t>(stream_id))
        << ", proc_load="
        << proc_loads.at(static_cast<std::size_t>(stream_id))
        << '\n';
  }

  std::cout << "===============================\n";
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

  /*
   * SM配分候補は00_pipeline_configuration.hppで定義済み。
   * Stage 1で得た最大並列幅から、評価するStream数の上限だけ決める。
   */
  const int stream_limit = decide_stream_count(
      analysis.levels,
      options
  );

  const std::vector<std::vector<int>> sm_count_candidates =
      make_sm_count_candidates(
          stream_limit,
          options
      );

  if (sm_count_candidates.empty()) {
    throw std::runtime_error("no SM allocation candidates");
  }

  /* GC使用時は候補が実機で実現可能か確認する。 */
  if (!options.disable_gc) {
    const SmPartitionInfo partition_info = query_sm_partition_info();
    validate_fixed_sm_table_compatibility(partition_info);

    for (const auto& sm_counts : sm_count_candidates) {
      validate_green_context_sm_counts(
          sm_counts,
          partition_info
      );
    }
  }

  /*
   * 各SM配分候補についてStage 2 -> Stage 3を個別に実行する。
   */
  std::vector<SmAllocationCandidate> all_candidates;
  all_candidates.reserve(sm_count_candidates.size());

  for (const auto& sm_counts : sm_count_candidates) {
    TaskImportanceResult importance;

    /* Stage 2: SM配分を考慮した予測bottom level */
    {
      ScopedTimer timer(benchmark.task_importance_ms);
      importance = evaluate_task_importance(
          analysis.tasks,
          sm_counts,
          kSchedulingReferenceSmCount
      );
    }

    StreamScheduleResult schedule;

    /* Stage 3: 予測完了時刻が最小のStreamへ配置 */
    {
      ScopedTimer timer(benchmark.stream_placement_ms);
      schedule = place_tasks_on_streams(
          analysis.tasks,
          importance,
          sm_counts,
          kSchedulingReferenceSmCount
      );
    }

    all_candidates.push_back(
        SmAllocationCandidate{
            sm_counts,
            std::move(importance),
            std::move(schedule)
        }
    );
  }

  /* Stage 4: 予測makespanが最小の実行構成を選択 */
  std::vector<SmAllocationCandidate> best_per_stream_count;
  SmAllocationDecision decision;

  {
    ScopedTimer timer(benchmark.sm_allocation_comparison_ms);

    best_per_stream_count =
        select_best_candidate_per_stream_count(all_candidates);

    decision = compare_sm_allocation_candidates(
        best_per_stream_count
    );
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
