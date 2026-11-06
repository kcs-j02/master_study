#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../proposed_method/00_pipeline_configuration.hpp"
#include "../../proposed_method/01_task_graph_analysis.hpp"
#include "../../proposed_method/02_task_importance.hpp"
#include "../../proposed_method/03_stream_scheduling.hpp"
#include "../../proposed_method/04_execution_configuration_selection.hpp"
#include "../../proposed_method/05_green_context_execution.cuh"
#include "../../proposed_method/bench_timer.hpp"

namespace {

enum class AblationMode {
  Full,
  NoImportance,
  NoSmAllocation,
  NoStreamPlacement,
};

AblationMode parse_mode(const char* value) {
  const std::string mode = value == nullptr ? "full" : value;
  if (mode == "full") return AblationMode::Full;
  if (mode == "no_importance") return AblationMode::NoImportance;
  if (mode == "no_sm_allocation") return AblationMode::NoSmAllocation;
  if (mode == "no_stream_placement") return AblationMode::NoStreamPlacement;
  throw std::invalid_argument(
      "STG_ABLATION_MODE must be full, no_importance, "
      "no_sm_allocation, or no_stream_placement");
}

int configured_stream_count() {
  const char* value = std::getenv("STG_ABLATION_STREAMS");
  if (value == nullptr) return 3;
  const int count = std::stoi(value);
  if (count < 1 || count > kMaximumConfiguredStreamCount) {
    throw std::invalid_argument("STG_ABLATION_STREAMS must be between 1 and 5");
  }
  return count;
}

TaskImportanceResult make_importance(
    const std::vector<TaskSpec>& tasks,
    AblationMode mode
) {
  TaskImportanceResult importance = evaluate_task_importance(tasks);
  if (mode == AblationMode::NoImportance) {
    for (auto& [task_id, bottom_level] : importance.bottom_levels) {
      static_cast<void>(task_id);
      bottom_level = 0.0;
    }
  }
  return importance;
}

std::vector<int> equal_sm_counts(int stream_count) {
  /*
   * Green ContextのSM配分は8 SM単位でなければならない。
   * したがって、ここでは均等化の基準となる固定候補
   * （追加チャンクの非均等最適化を行う前）を使用する。
   */
  const auto candidates = make_stream_sm_count_candidates(stream_count);
  return candidates.front();
}

BenchResult run(const std::string& path, AblationMode mode) {
  BenchResult result;
  const auto total_start = Clock::now();
  const int stream_count = configured_stream_count();

  StgAnalysisResult analysis;
  {
    ScopedTimer timer(result.stg_analysis_ms);
    analysis = analyze_stg(path);
  }

  TaskImportanceResult importance;
  {
    ScopedTimer timer(result.task_importance_ms);
    importance = make_importance(analysis.tasks, mode);
  }

  StreamScheduleResult schedule;
  {
    ScopedTimer timer(result.stream_placement_ms);
    if (mode == AblationMode::NoStreamPlacement) {
      schedule = place_tasks_on_streams(
          analysis.tasks, importance, 1);
    } else {
      schedule = place_tasks_on_streams(
          analysis.tasks, importance, stream_count);
    }
  }

  std::vector<int> sm_counts;
  bool disable_green_context = false;
  {
    ScopedTimer timer(result.sm_allocation_ms);
    if (mode == AblationMode::NoStreamPlacement) {
      sm_counts = {kSchedulingReferenceSmCount};
      disable_green_context = true;
    } else if (mode == AblationMode::NoSmAllocation) {
      sm_counts = equal_sm_counts(stream_count);
      disable_green_context = false;
    } else {
      const auto candidates = make_stream_sm_count_candidates(stream_count);
      const auto allocation = allocate_sms_for_placement(
          analysis.tasks, schedule, candidates.front(), true);
      sm_counts = allocation.sm_counts;
    }
  }

  GreenContextExecutionResult execution;
  {
    ScopedTimer timer(result.green_context_execution_ms);
    execution = execute_with_green_context(
        analysis.tasks,
        schedule,
        sm_counts,
        GreenContextExecutionOptions{
            disable_green_context,
            1,
            kSchedulingReferenceSmCount});
  }

  result.gpu_submit_wait_ms = execution.gpu_submit_wait_ms;
  result.gpu_kernel_ms = execution.gpu_kernel_ms;
  result.total_ms = elapsed_ms(total_start, Clock::now());
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " input.stg\n";
    return 1;
  }

  try {
    const auto mode = parse_mode(std::getenv("STG_ABLATION_MODE"));
    const BenchResult result = run(argv[1], mode);
    std::cout << "mode: strict_ablation\n";
    std::cout << "ablation_mode: "
              << (std::getenv("STG_ABLATION_MODE") == nullptr
                      ? "full"
                      : std::getenv("STG_ABLATION_MODE"))
              << '\n';
    std::cout << "ablation_streams: " << configured_stream_count() << '\n';
    print_result(result);
  } catch (const std::exception& error) {
    std::cerr << "exception: " << error.what() << '\n';
    return 1;
  }
}
