// kobayashi@h100:~/main/STG$ cat > sample.stg << 'EOF'
// 6
// 0 0 0
// 1 10 1 0
// 2 30 1 0
// 3 5 1 1
// 4 40 2 1 2
// 5 0 2 3 4
// EOF

// 実行はこれ！！！！
// nvcc -O2 -std=c++20 taskflow_stg_cuda_demo.cu -I/home/kobayashi/taskflow -o taskflow_stg_cuda_demo
// ./taskflow_stg_cuda_demo sample.stg baseline
// ./taskflow_stg_cuda_demo sample.stg aware

// nsys profile ./taskflow_stg_cuda_demo sample.stg baseline

// kobayashi@h100:~/main/STG$ nsys profile \
//   --trace=cuda,nvtx,osrt \
//   --gpu-metrics-devices=0 \
//   --gpu-metrics-frequency=10000 \
//   -o stg_baseline \
//   ./taskflow_stg_cuda_demo sample.stg baseline

#include <cuda_runtime.h>
#include <taskflow/taskflow.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "bench_timer.hpp"

// ============================================================
// CUDA utilities
// ============================================================

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _err = (expr);                                                \
    if (_err != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                 \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// ============================================================
// STG data model
// current STG without communication costs
// line1: N
// next N lines: task_id proc_time pred_count pred1 pred2 ...
// lines beginning with # are info/comments
// ============================================================

struct StgTask {
  int id = -1;
  int proc_time = 0;
  std::vector<int> preds;
};

struct StgGraph {
  int num_tasks = 0;
  std::vector<StgTask> tasks;
};

static inline std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";

  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

StgGraph load_stg_without_comm(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("failed to open STG file: " + path);
  }

  std::vector<std::string> raw_lines;
  std::string line;

  while (std::getline(ifs, line)) {
    line = trim(line);
    if (line.empty()) continue;
    if (!line.empty() && line[0] == '#') continue;

    raw_lines.push_back(line);
  }

  if (raw_lines.empty()) {
    throw std::runtime_error("empty STG file: " + path);
  }

  StgGraph g;

  {
    std::istringstream iss(raw_lines[0]);
    if (!(iss >> g.num_tasks)) {
      throw std::runtime_error("failed to parse number of tasks");
    }
  }

  if (static_cast<int>(raw_lines.size()) < 1 + g.num_tasks) {
    throw std::runtime_error("STG file has fewer task lines than expected");
  }

  g.tasks.reserve(g.num_tasks);

  for (int i = 0; i < g.num_tasks; ++i) {
    std::istringstream iss(raw_lines[1 + i]);

    StgTask t;
    int pred_count = 0;

    if (!(iss >> t.id >> t.proc_time >> pred_count)) {
      throw std::runtime_error(
        "failed to parse task header at task-line index " + std::to_string(i)
      );
    }

    if (pred_count < 0) {
      throw std::runtime_error(
        "negative predecessor count at task " + std::to_string(t.id)
      );
    }

    t.preds.resize(pred_count);

    for (int k = 0; k < pred_count; ++k) {
      if (!(iss >> t.preds[k])) {
        throw std::runtime_error(
          "failed to parse predecessor list at task " + std::to_string(t.id)
        );
      }
    }

    g.tasks.push_back(std::move(t));
  }

  // task id uniqueness check
  std::vector<int> ids;
  ids.reserve(g.tasks.size());

  for (const auto& t : g.tasks) {
    ids.push_back(t.id);
  }

  std::sort(ids.begin(), ids.end());

  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
    throw std::runtime_error("duplicate task ids found in STG");
  }

  return g;
}

// ============================================================
// Example resource classification
// heavy/light split from processing time
// ============================================================

enum class KernelKind {
  LIGHT,
  HEAVY
};

enum class ResourceClass {
  SMALL,
  LARGE
};

struct TaskSpec {
  int id = -1;
  int proc_time = 0;

  KernelKind kind = KernelKind::LIGHT;
  ResourceClass rclass = ResourceClass::SMALL;

  int work_units = 1000;
  std::vector<int> preds;
};

std::vector<TaskSpec> make_task_specs_from_stg(const StgGraph& g) {
  std::vector<int> positive_times;
  positive_times.reserve(g.tasks.size());

  for (const auto& t : g.tasks) {
    if (t.proc_time > 0) {
      positive_times.push_back(t.proc_time);
    }
  }

  int threshold = 1;

  if (!positive_times.empty()) {
    std::sort(positive_times.begin(), positive_times.end());
    threshold = positive_times[positive_times.size() / 2];  // median
  }

  std::vector<TaskSpec> specs;
  specs.reserve(g.tasks.size());

  for (const auto& t : g.tasks) {
    TaskSpec s;

    s.id = t.id;
    s.proc_time = t.proc_time;
    s.preds = t.preds;

    // entry/exit dummy など proc_time=0 は軽い扱い
    if (t.proc_time > threshold) {
      s.kind = KernelKind::HEAVY;
      s.rclass = ResourceClass::LARGE;
      s.work_units = std::max(1000, t.proc_time * 200);
    } else {
      s.kind = KernelKind::LIGHT;
      s.rclass = ResourceClass::SMALL;
      s.work_units = std::max(200, std::max(1, t.proc_time) * 80);
    }

    specs.push_back(std::move(s));
  }

  return specs;
}

// ============================================================
// CUDA kernels
// ============================================================

__global__ void light_kernel(float* data, int n, int iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float x = data[idx];

    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
      x = x * 1.000001f + 0.00001f;
    }

    data[idx] = x;
  }
}

__global__ void heavy_kernel(float* data, int n, int iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float x = data[idx];

    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
      x = x * 1.000001f + 0.00001f;
      x = x * 0.999999f + 0.00002f;
      x = x * 1.0000003f - 0.00001f;
    }

    data[idx] = x;
  }
}

void launch_task_kernel(const TaskSpec& task,
                        float* dmem,
                        int n,
                        cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);

  if (task.kind == KernelKind::HEAVY) {
    heavy_kernel<<<grid, block, 0, stream>>>(dmem, n, task.work_units);
  } else {
    light_kernel<<<grid, block, 0, stream>>>(dmem, n, task.work_units);
  }

  CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// resource allocation abstraction
// 今は stream 分離
// 本物の Green Context を差し込むならここを置換
// ============================================================

struct PartitionContext {
  cudaStream_t stream {};
  std::string label;
};

struct RuntimeResources {
  PartitionContext baseline0;
  PartitionContext baseline1;

  PartitionContext small0;
  PartitionContext small1;

  PartitionContext large0;
  PartitionContext large1;
};

RuntimeResources make_runtime_resources() {
  RuntimeResources rr;

  CUDA_CHECK(cudaStreamCreate(&rr.baseline0.stream));
  rr.baseline0.label = "baseline0";

  CUDA_CHECK(cudaStreamCreate(&rr.baseline1.stream));
  rr.baseline1.label = "baseline1";

  CUDA_CHECK(cudaStreamCreate(&rr.small0.stream));
  rr.small0.label = "small0";

  CUDA_CHECK(cudaStreamCreate(&rr.small1.stream));
  rr.small1.label = "small1";

  CUDA_CHECK(cudaStreamCreate(&rr.large0.stream));
  rr.large0.label = "large0";

  CUDA_CHECK(cudaStreamCreate(&rr.large1.stream));
  rr.large1.label = "large1";

  return rr;
}

void destroy_runtime_resources(RuntimeResources& rr) {
  CUDA_CHECK(cudaStreamDestroy(rr.baseline0.stream));
  CUDA_CHECK(cudaStreamDestroy(rr.baseline1.stream));

  CUDA_CHECK(cudaStreamDestroy(rr.small0.stream));
  CUDA_CHECK(cudaStreamDestroy(rr.small1.stream));

  CUDA_CHECK(cudaStreamDestroy(rr.large0.stream));
  CUDA_CHECK(cudaStreamDestroy(rr.large1.stream));
}

cudaStream_t select_baseline_stream(RuntimeResources& rr) {
  static std::atomic<int> turn{0};

  int v = turn.fetch_add(1, std::memory_order_relaxed);
  return (v % 2 == 0) ? rr.baseline0.stream : rr.baseline1.stream;
}

cudaStream_t select_partitioned_stream(const TaskSpec& t,
                                       RuntimeResources& rr) {
  static std::atomic<int> small_turn{0};
  static std::atomic<int> large_turn{0};

  if (t.rclass == ResourceClass::LARGE) {
    int v = large_turn.fetch_add(1, std::memory_order_relaxed);
    return (v % 2 == 0) ? rr.large0.stream : rr.large1.stream;
  }

  int v = small_turn.fetch_add(1, std::memory_order_relaxed);
  return (v % 2 == 0) ? rr.small0.stream : rr.small1.stream;
}

// ============================================================
// execution
// ============================================================

BenchResult run_taskflow_cuda(const std::vector<TaskSpec>& tasks,
                              bool resource_aware) {
  BenchResult bench;

  auto total_start = Clock::now();

  constexpr int N = 1 << 20;

  float* dmem = nullptr;

  CUDA_CHECK(cudaMalloc(&dmem, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(dmem, 0, N * sizeof(float)));

  RuntimeResources rr = make_runtime_resources();

  tf::Executor executor(8);
  tf::Taskflow tf;

  // task id -> tf::Task
  std::unordered_map<int, tf::Task> task_nodes;

  // task id -> completion event
  std::unordered_map<int, std::unique_ptr<cudaEvent_t>> done_events;

  // task id -> kernel timing event
  std::unordered_map<int, cudaEvent_t> kernel_start_events;
  std::unordered_map<int, cudaEvent_t> kernel_stop_events;

  // ----------------------------------------
  // 1. 各 task 用の CUDA event を先に作る
  // ----------------------------------------
  for (const auto& t : tasks) {
    done_events[t.id] = std::make_unique<cudaEvent_t>();

    CUDA_CHECK(cudaEventCreateWithFlags(
      done_events[t.id].get(),
      cudaEventDisableTiming
    ));

    cudaEvent_t start_ev;
    cudaEvent_t stop_ev;

    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    kernel_start_events.emplace(t.id, start_ev);
    kernel_stop_events.emplace(t.id, stop_ev);
  }

  // ----------------------------------------
  // 2. STG の各 task から tf::Task を作る
  // ----------------------------------------
  for (const auto& t : tasks) {
    tf::Task node = tf.emplace([&, t]() {
      cudaStream_t stream = resource_aware
        ? select_partitioned_stream(t, rr)
        : select_baseline_stream(rr);

      // 先行 task の完了を CUDA stream 上で待つ
      for (int pred : t.preds) {
        auto it = done_events.find(pred);

        if (it == done_events.end()) {
          throw std::runtime_error(
            "unknown predecessor id: " + std::to_string(pred)
          );
        }

        CUDA_CHECK(cudaStreamWaitEvent(stream, *(it->second), 0));
      }

      // kernel 開始 event
      CUDA_CHECK(cudaEventRecord(kernel_start_events.at(t.id), stream));

      // この task の kernel を launch
      launch_task_kernel(t, dmem, N, stream);

      // kernel 終了 event
      CUDA_CHECK(cudaEventRecord(kernel_stop_events.at(t.id), stream));

      // この task の完了 event を記録
      CUDA_CHECK(cudaEventRecord(*(done_events.at(t.id)), stream));
    });

    node.name(std::to_string(t.id));
    task_nodes.emplace(t.id, node);
  }

  // ----------------------------------------
  // 3. STG の依存関係 pred -> task を tf に張る
  // ----------------------------------------
  for (const auto& t : tasks) {
    for (int pred : t.preds) {
      auto pred_it = task_nodes.find(pred);
      auto curr_it = task_nodes.find(t.id);

      if (pred_it == task_nodes.end() || curr_it == task_nodes.end()) {
        throw std::runtime_error("edge references unknown task id");
      }

      pred_it->second.precede(curr_it->second);
    }
  }

  // ----------------------------------------
  // 4. GPU submit + wait の時間を測定
  // ----------------------------------------
  auto submit_start = Clock::now();

  executor.run(tf).wait();
  CUDA_CHECK(cudaDeviceSynchronize());

  auto submit_end = Clock::now();

  bench.gpu_submit_wait_ms = elapsed_ms(submit_start, submit_end);

  // ----------------------------------------
  // 5. GPU kernel 時間を集計
  // ----------------------------------------
  double kernel_sum_ms = 0.0;

  for (const auto& t : tasks) {
    float ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
      &ms,
      kernel_start_events.at(t.id),
      kernel_stop_events.at(t.id)
    ));

    kernel_sum_ms += static_cast<double>(ms);
  }

  bench.gpu_kernel_ms = kernel_sum_ms;

  // ----------------------------------------
  // 6. 後始末
  // ----------------------------------------
  for (auto& [id, ev] : done_events) {
    CUDA_CHECK(cudaEventDestroy(*ev));
  }

  for (auto& [id, ev] : kernel_start_events) {
    CUDA_CHECK(cudaEventDestroy(ev));
  }

  for (auto& [id, ev] : kernel_stop_events) {
    CUDA_CHECK(cudaEventDestroy(ev));
  }

  destroy_runtime_resources(rr);

  CUDA_CHECK(cudaFree(dmem));

  auto total_end = Clock::now();

  bench.total_ms = elapsed_ms(total_start, total_end);

  // 今回この4項目は測定しないので 0.0 のまま
  bench.task_DFG_construction_ms = 0.0;
  bench.task_levelization_ms = 0.0;
  bench.task_assignment_ms = 0.0;
  bench.resource_allocation_ms = 0.0;

  return bench;
}

// ============================================================
// dump summary
// ============================================================

void print_stg_summary(const std::vector<TaskSpec>& tasks) {
  int heavy = 0;
  int light = 0;
  long long total_proc = 0;

  for (const auto& t : tasks) {
    total_proc += t.proc_time;

    if (t.kind == KernelKind::HEAVY) {
      ++heavy;
    } else {
      ++light;
    }
  }

  std::cout << "===== STG summary =====\n";
  std::cout << "num_tasks     : " << tasks.size() << "\n";
  std::cout << "light_tasks   : " << light << "\n";
  std::cout << "heavy_tasks   : " << heavy << "\n";
  std::cout << "sum_proc_time : " << total_proc << "\n";
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " input.stg [baseline|aware]\n";
    return 1;
  }

  try {
    // 1. STGファイルを読み込む
    StgGraph g = load_stg_without_comm(argv[1]);

    // 2. STGを TaskSpec の配列に変換する
    auto specs = make_task_specs_from_stg(g);

    print_stg_summary(specs);

    std::string mode = argv[2];

    if (mode == "baseline") {
      BenchResult r = run_taskflow_cuda(specs, false);

      std::cout << "mode: baseline\n";
      print_result(r);
    }
    else if (mode == "aware") {
      BenchResult r = run_taskflow_cuda(specs, true);

      std::cout << "mode: resource-aware\n";
      print_result(r);
    }
    else {
      std::cerr << "mode must be baseline or aware\n";
      return 1;
    }
  }
  catch (const std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
  }

  return 0;
}