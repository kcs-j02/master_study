#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../benchmarks/common/task_graph_common.hpp"
#include "../../proposed_method/05_green_context_execution.cuh"

namespace {

struct Series {
  std::string benchmark;
  int task_id = -1;
  std::string kernel;
  int sm_count = 0;
  std::vector<float> samples_ms;
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

double median(std::vector<float> values) {
  if (values.empty()) throw std::invalid_argument("median requires samples");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 != 0) return values.at(middle);
  return (static_cast<double>(values.at(middle - 1)) + values.at(middle)) / 2.0;
}

std::string kernel_name(stg::BenchmarkKind kind, int task_id) {
  switch (kind) {
    case stg::BenchmarkKind::VECTOR_SQUARE:
      return task_id < 2 ? "vector_square" : "vector_reduce";
    case stg::BenchmarkKind::BLACK_SCHOLES:
      return "black_scholes";
    case stg::BenchmarkKind::MACHINE_LEARNING: {
      static const std::array<const char*, 10> names = {
          "ridge_scale", "naive_bayes_matmul", "ridge_matmul", "row_max",
          "logsumexp", "add_intercept", "normalize", "softmax_nb",
          "softmax_ridge", "argmax"};
      return names.at(static_cast<std::size_t>(task_id));
    }
    case stg::BenchmarkKind::HITS: {
      static const std::array<const char*, 7> names = {
          "spmv_authority", "spmv_hub", "sum_authority", "sum_hub",
          "divide_authority", "divide_hub", "reset"};
      return names.at(static_cast<std::size_t>(task_id % 7));
    }
    case stg::BenchmarkKind::IMAGE_PROCESSING: {
      static const std::array<const char*, 11> names = {
          "blur_small", "blur_large", "blur_unsharpen", "sobel_small",
          "sobel_large", "maximum", "minimum", "extend", "unsharpen",
          "combine_image2", "combine_image3"};
      return names.at(static_cast<std::size_t>(task_id));
    }
    case stg::BenchmarkKind::DEEP_LEARNING: {
      static const std::array<const char*, 8> names = {
          "conv_layer1_x", "conv_layer1_y", "mean_pool_x", "mean_pool_y",
          "conv_layer2_x", "conv_layer2_y", "concat", "dot_product"};
      return names.at(static_cast<std::size_t>(task_id));
    }
    case stg::BenchmarkKind::SYNTHETIC:
      return "synthetic";
  }
  throw std::logic_error("unknown benchmark kind");
}

std::string benchmark_name(stg::BenchmarkKind kind) {
  switch (kind) {
    case stg::BenchmarkKind::VECTOR_SQUARE: return "vector_square";
    case stg::BenchmarkKind::BLACK_SCHOLES: return "black_scholes";
    case stg::BenchmarkKind::MACHINE_LEARNING: return "machine_learning";
    case stg::BenchmarkKind::HITS: return "hits";
    case stg::BenchmarkKind::IMAGE_PROCESSING: return "image_processing";
    case stg::BenchmarkKind::DEEP_LEARNING: return "deep_learning";
    case stg::BenchmarkKind::SYNTHETIC: break;
  }
  throw std::logic_error("synthetic input is not supported");
}

std::vector<stg::TaskSpec> make_tasks(const stg::StgGraph& graph) {
  std::vector<stg::TaskSpec> tasks;
  tasks.reserve(graph.tasks.size());
  for (const auto& input : graph.tasks) {
    stg::TaskSpec task;
    task.id = input.id;
    task.proc_time = input.proc_time;
    task.preds = input.preds;
    task.benchmark_kind = graph.benchmark_kind;
    task.parallel_sm_limit = kTaskParallelSmLimit;
    tasks.push_back(std::move(task));
  }
  return tasks;
}

// Restore only data that a benchmark mutates and later reuses. This runs and
// synchronizes before timing starts, so initialization is excluded.
void reset_workspace(stg::kesco::Workspace& w) {
  using namespace stg::kesco;
  switch (w.kind) {
    case stg::BenchmarkKind::VECTOR_SQUARE:
      check_cuda(cudaMemset(w.f[4], 0, sizeof(float)), "reset vector reduction");
      break;
    case stg::BenchmarkKind::BLACK_SCHOLES:
      break;
    case stg::BenchmarkKind::MACHINE_LEARNING:
      check_cuda(cudaMemset(w.f[6], 0,
          static_cast<std::size_t>(w.n) * 10 * sizeof(float)), "reset ML output 1");
      check_cuda(cudaMemset(w.f[7], 0,
          static_cast<std::size_t>(w.n) * 10 * sizeof(float)), "reset ML output 2");
      break;
    case stg::BenchmarkKind::HITS:
      initialize(w.f[0], w.n, 1.0f);
      initialize(w.f[2], w.n, 1.0f);
      check_cuda(cudaMemset(w.f[4], 0, sizeof(float)), "reset HITS authority sum");
      check_cuda(cudaMemset(w.f[5], 0, sizeof(float)), "reset HITS hub sum");
      break;
    case stg::BenchmarkKind::IMAGE_PROCESSING: {
      const float maximum = -1000.0f;
      const float minimum = 1000.0f;
      check_cuda(cudaMemcpy(w.f[13], &maximum, sizeof(float), cudaMemcpyHostToDevice),
                 "reset image maximum");
      check_cuda(cudaMemcpy(w.f[14], &minimum, sizeof(float), cudaMemcpyHostToDevice),
                 "reset image minimum");
      break;
    }
    case stg::BenchmarkKind::DEEP_LEARNING:
      check_cuda(cudaMemset(w.f[14], 0, sizeof(float)), "reset DL dot product");
      break;
    case stg::BenchmarkKind::SYNTHETIC:
      throw std::logic_error("synthetic workspace is not supported");
  }
  check_cuda(cudaDeviceSynchronize(), "synchronize reset");
}

void launch_graph(const std::vector<stg::TaskSpec>& tasks,
                  const std::vector<std::size_t>& order,
                  stg::kesco::Workspace& workspace,
                  cudaStream_t stream) {
  for (const std::size_t index : order) {
    stg::kesco::launch(tasks.at(index), workspace, stream);
  }
}

RuntimeResources make_resources(int available_sm, int sm_count) {
  return sm_count == available_sm
      ? create_green_context_resources({available_sm})
      : create_green_context_resources({available_sm - sm_count, sm_count});
}

cudaStream_t measured_stream(RuntimeResources& resources, int available_sm,
                             int sm_count) {
  const std::size_t stream_id = sm_count == available_sm ? 0 : 1;
  return resources.streams.at(stream_id);
}

Series measure_total(const std::string& name,
                     const std::vector<stg::TaskSpec>& tasks,
                     const std::vector<std::size_t>& order,
                     stg::kesco::Workspace& workspace,
                     int available_sm,
                     int sm_count,
                     int warmups,
                     int runs) {
  RuntimeResources resources = make_resources(available_sm, sm_count);
  const cudaStream_t stream = measured_stream(resources, available_sm, sm_count);
  for (int run = 0; run < warmups; ++run) {
    reset_workspace(workspace);
    launch_graph(tasks, order, workspace, stream);
    check_cuda(cudaStreamSynchronize(stream), "total warmup synchronization");
  }

  cudaEvent_t start = nullptr, stop = nullptr;
  check_cuda(cudaEventCreate(&start), "create total start event");
  check_cuda(cudaEventCreate(&stop), "create total stop event");
  Series result{name, -1, "ALL", sm_count, {}};
  result.samples_ms.reserve(static_cast<std::size_t>(runs));
  for (int run = 0; run < runs; ++run) {
    reset_workspace(workspace);
    check_cuda(cudaEventRecord(start, stream), "record total start");
    launch_graph(tasks, order, workspace, stream);
    check_cuda(cudaEventRecord(stop, stream), "record total stop");
    check_cuda(cudaEventSynchronize(stop), "wait total stop");
    float elapsed = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "read total time");
    result.samples_ms.push_back(elapsed);
  }
  check_cuda(cudaEventDestroy(start), "destroy total start event");
  check_cuda(cudaEventDestroy(stop), "destroy total stop event");
  destroy_runtime_resources(resources);
  return result;
}

std::vector<Series> measure_tasks(const std::string& name,
                                  const std::vector<stg::TaskSpec>& tasks,
                                  const std::vector<std::size_t>& order,
                                  stg::kesco::Workspace& workspace,
                                  int available_sm,
                                  int sm_count,
                                  int warmups,
                                  int runs) {
  RuntimeResources resources = make_resources(available_sm, sm_count);
  const cudaStream_t stream = measured_stream(resources, available_sm, sm_count);
  for (int run = 0; run < warmups; ++run) {
    reset_workspace(workspace);
    launch_graph(tasks, order, workspace, stream);
    check_cuda(cudaStreamSynchronize(stream), "task warmup synchronization");
  }

  std::vector<Series> results;
  results.reserve(tasks.size());
  for (const auto& task : tasks) {
    results.push_back(Series{name, task.id,
        kernel_name(task.benchmark_kind, task.id), sm_count, {}});
    results.back().samples_ms.reserve(static_cast<std::size_t>(runs));
  }

  cudaEvent_t start = nullptr, stop = nullptr;
  check_cuda(cudaEventCreate(&start), "create task start event");
  check_cuda(cudaEventCreate(&stop), "create task stop event");
  for (int run = 0; run < runs; ++run) {
    reset_workspace(workspace);
    for (const std::size_t index : order) {
      check_cuda(cudaEventRecord(start, stream), "record task start");
      stg::kesco::launch(tasks.at(index), workspace, stream);
      check_cuda(cudaEventRecord(stop, stream), "record task stop");
      check_cuda(cudaEventSynchronize(stop), "wait task stop");
      float elapsed = 0.0f;
      check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "read task time");
      results.at(index).samples_ms.push_back(elapsed);
    }
  }
  check_cuda(cudaEventDestroy(start), "destroy task start event");
  check_cuda(cudaEventDestroy(stop), "destroy task stop event");
  destroy_runtime_resources(resources);
  return results;
}

void write_csv(const std::filesystem::path& path,
               const std::vector<Series>& all_series,
               int warmups,
               int runs,
               int reference_sm) {
  std::map<std::pair<std::string, std::pair<int, std::string>>, double> reference;
  for (const auto& series : all_series) {
    if (series.sm_count == reference_sm) {
      reference[{series.benchmark, {series.task_id, series.kernel}}] =
          median(series.samples_ms);
    }
  }
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open output CSV: " + path.string());
  output << "benchmark,scope,task_id,kernel,sm_count,warmup_runs,measured_runs,"
            "median_ms,min_ms,max_ms,slowdown_vs_full_gpu\n";
  output << std::fixed << std::setprecision(9);
  for (const auto& series : all_series) {
    const double med = median(series.samples_ms);
    const auto key = std::make_pair(series.benchmark,
                                    std::make_pair(series.task_id, series.kernel));
    output << series.benchmark << ','
           << (series.task_id < 0 ? "benchmark" : "task") << ','
           << series.task_id << ',' << series.kernel << ',' << series.sm_count << ','
           << warmups << ',' << runs << ',' << med << ','
           << *std::min_element(series.samples_ms.begin(), series.samples_ms.end()) << ','
           << *std::max_element(series.samples_ms.begin(), series.samples_ms.end()) << ','
           << med / reference.at(key) << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::filesystem::path study_dir =
        std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
    const std::filesystem::path output = argc >= 2
        ? std::filesystem::path(argv[1])
        : study_dir / "results/method_comparison_figures/kesco_sm_scaling/kesco_sm_scaling.csv";
    const int warmups = argc >= 3 ? std::stoi(argv[2]) : 5;
    const int runs = argc >= 4 ? std::stoi(argv[3]) : 21;
    if (warmups < 0 || runs <= 0) throw std::invalid_argument("invalid run counts");

    const SmPartitionInfo partition = query_sm_partition_info();
    validate_fixed_sm_table_compatibility(partition);
    std::vector<int> sm_counts;
    for (int sm = partition.min_group_sm; sm < partition.available_sm;
         sm += partition.unit_sm) sm_counts.push_back(sm);
    sm_counts.push_back(partition.available_sm);

    const std::array<std::string, 6> names = {
        "vector_square", "black_scholes", "machine_learning", "hits",
        "image_processing", "deep_learning"};
    std::vector<Series> all_series;
    for (const auto& expected_name : names) {
      const auto stg_path = study_dir / "benchmarks/KESCO" /
          ("BENCHMARK_" + expected_name) / (expected_name + ".stg");
      const stg::StgGraph graph = stg::load_stg_without_comm(stg_path.string());
      const std::string name = benchmark_name(graph.benchmark_kind);
      const auto tasks = make_tasks(graph);
      const auto order = stg::make_topological_order(tasks);
      auto workspace = stg::kesco::create_workspace(graph.benchmark_kind);
      std::cout << "\n[" << name << "] tasks=" << tasks.size() << std::endl;
      for (const int sm_count : sm_counts) {
        std::cout << "  SM=" << std::setw(3) << sm_count << std::flush;
        all_series.push_back(measure_total(name, tasks, order, workspace,
                                           partition.available_sm, sm_count,
                                           warmups, runs));
        auto task_series = measure_tasks(name, tasks, order, workspace,
                                         partition.available_sm, sm_count,
                                         warmups, runs);
        all_series.insert(all_series.end(),
                          std::make_move_iterator(task_series.begin()),
                          std::make_move_iterator(task_series.end()));
        std::cout << " total_median_ms=" << std::setprecision(6) << std::fixed
                  << median(all_series.at(all_series.size() - tasks.size() - 1).samples_ms)
                  << std::endl;
      }
      stg::kesco::destroy_workspace(workspace);
    }
    write_csv(output, all_series, warmups, runs, partition.available_sm);
    std::cout << "\nresult: " << output << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
