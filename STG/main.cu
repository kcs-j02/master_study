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
// nvcc -O2 -std=c++20 main.cu -I/home/kobayashi/taskflow -o main 
// ./main sample.stg baseline

// nsys profile ./main sample.stg baseline

// GPU使用率
// nsys profile \
//   --force-overwrite true \
//   --trace=cuda,nvtx,osrt \
//   --gpu-metrics-devices=0 \
//   --gpu-metrics-frequency=10000 \
//   -o stg_profile \
//   ./main sample.stg

#include <cuda_runtime.h>
#include <taskflow/taskflow.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "bench_timer.hpp"

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _err = (expr);                                                \
    if (_err != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                 \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

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

enum class KernelKind {
  LIGHT,
  HEAVY
};

struct TaskSpec {
  int id = -1;
  int proc_time = 0;
  KernelKind kind = KernelKind::LIGHT;
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
    threshold = positive_times[positive_times.size() / 2];
  }

  std::vector<TaskSpec> specs;
  specs.reserve(g.tasks.size());

  for (const auto& t : g.tasks) {
    TaskSpec s;

    s.id = t.id;
    s.proc_time = t.proc_time;
    s.preds = t.preds;

    if (t.proc_time > threshold) {
      s.kind = KernelKind::HEAVY;
      s.work_units = std::max(1000, t.proc_time * 200);
    } else {
      s.kind = KernelKind::LIGHT;
      s.work_units = std::max(200, std::max(1, t.proc_time) * 80);
    }

    specs.push_back(std::move(s));
  }

  return specs;
}

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

struct RuntimeResources {
  cudaStream_t baseline0 {};
  cudaStream_t baseline1 {};
};

RuntimeResources make_runtime_resources() {
  RuntimeResources rr;

  CUDA_CHECK(cudaStreamCreate(&rr.baseline0));
  CUDA_CHECK(cudaStreamCreate(&rr.baseline1));

  return rr;
}

void destroy_runtime_resources(RuntimeResources& rr) {
  CUDA_CHECK(cudaStreamDestroy(rr.baseline0));
  CUDA_CHECK(cudaStreamDestroy(rr.baseline1));
}

cudaStream_t select_baseline_stream(RuntimeResources& rr) {
  static std::atomic<int> turn{0};

  int v = turn.fetch_add(1, std::memory_order_relaxed);
  return (v % 2 == 0) ? rr.baseline0 : rr.baseline1;
}

BenchResult run_taskflow_cuda(const std::vector<TaskSpec>& tasks) {
  BenchResult bench;

  auto total_start = Clock::now();

  constexpr int N = 1 << 20;

  float* dmem = nullptr;

  CUDA_CHECK(cudaMalloc(&dmem, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(dmem, 0, N * sizeof(float)));

  RuntimeResources rr = make_runtime_resources();

  tf::Executor executor(8);
  tf::Taskflow tf;

  std::unordered_map<int, tf::Task> task_nodes;
  std::unordered_map<int, cudaEvent_t> done_events;
  std::unordered_map<int, cudaEvent_t> kernel_start_events;
  std::unordered_map<int, cudaEvent_t> kernel_stop_events;

  for (const auto& t : tasks) {
    cudaEvent_t done_ev;
    cudaEvent_t start_ev;
    cudaEvent_t stop_ev;

    CUDA_CHECK(cudaEventCreateWithFlags(&done_ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    done_events.emplace(t.id, done_ev);
    kernel_start_events.emplace(t.id, start_ev);
    kernel_stop_events.emplace(t.id, stop_ev);
  }

  for (const auto& t : tasks) {
    tf::Task node = tf.emplace([&, t]() {
      cudaStream_t stream = select_baseline_stream(rr);

      for (int pred : t.preds) {
        auto it = done_events.find(pred);

        if (it == done_events.end()) {
          throw std::runtime_error(
            "unknown predecessor id: " + std::to_string(pred)
          );
        }

        CUDA_CHECK(cudaStreamWaitEvent(stream, it->second, 0));
      }

      CUDA_CHECK(cudaEventRecord(kernel_start_events.at(t.id), stream));

      launch_task_kernel(t, dmem, N, stream);

      CUDA_CHECK(cudaEventRecord(kernel_stop_events.at(t.id), stream));
      CUDA_CHECK(cudaEventRecord(done_events.at(t.id), stream));
    });

    node.name(std::to_string(t.id));
    task_nodes.emplace(t.id, node);
  }

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

  auto submit_start = Clock::now();

  executor.run(tf).wait();
  CUDA_CHECK(cudaDeviceSynchronize());

  auto submit_end = Clock::now();

  bench.gpu_submit_wait_ms = elapsed_ms(submit_start, submit_end);

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

  for (auto& [id, ev] : done_events) {
    CUDA_CHECK(cudaEventDestroy(ev));
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

  bench.task_DFG_construction_ms = 0.0;
  bench.task_levelization_ms = 0.0;
  bench.task_assignment_ms = 0.0;
  bench.resource_allocation_ms = 0.0;

  return bench;
}

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

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " input.stg\n";
    return 1;
  }

  try {
    StgGraph g = load_stg_without_comm(argv[1]);

    auto specs = make_task_specs_from_stg(g);

    print_stg_summary(specs);

    BenchResult r = run_taskflow_cuda(specs);

    std::cout << "mode: baseline\n";
    print_result(r);
  }
  catch (const std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
  }

  return 0;
}