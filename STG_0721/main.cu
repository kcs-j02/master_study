

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
// rm -rf main
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

// mkdir -p $HOME/tmp/nsys
// TMPDIR=$HOME/tmp/nsys nsys profile \
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
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "bench_timer.hpp"

// 追加
#include "common_types.hpp"
#include "task_DFG_construction.hpp"
#include "task_levelization.hpp"
#include "task_assignment.hpp"
#include "resource_allocation.hpp"

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _err = (expr);                                                \
    if (_err != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                 \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)



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

// 全タスクを単一ストリームで逐次実行した場合のGPU実行時間を計測する。
// メモリ確保・初期化・イベント生成などの時間は GPU_KERNEL_MS に含めない。
double measure_sequential_kernel_ms(const std::vector<TaskSpec>& tasks, int n) {
  float* dmem = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;

  CUDA_CHECK(cudaMalloc(&dmem, n * sizeof(float)));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  // 初期化処理は計測対象外にする。
  CUDA_CHECK(cudaMemsetAsync(dmem, 0, n * sizeof(float), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaEventRecord(start_event, stream));

  // 同一ストリームへの投入なので、tasks の順番で必ず逐次実行される。
  for (const auto& task : tasks) {
    launch_task_kernel(task, dmem, n, stream);
  }

  CUDA_CHECK(cudaEventRecord(stop_event, stream));
  CUDA_CHECK(cudaEventSynchronize(stop_event));

  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event, stop_event));

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dmem));

  return static_cast<double>(elapsed);
}






// 実行
BenchResult run_taskflow_cuda(const std::vector<TaskSpec>& tasks) {
  BenchResult bench;

  auto total_start = Clock::now();
  
  // 追加
  TaskDFG dfg;
  {
    ScopedTimer timer(bench.task_DFG_construction_ms);
    dfg = task_DFG_construction(tasks);
  }

  TaskLevels levels;

  {
    ScopedTimer timer(bench.task_levelization_ms);
    levels = task_levelization(dfg);
  }

  int stream_count;

  {
    ScopedTimer timer(bench.task_assignment_ms);
    
    constexpr int max_stream_count = 5;
    
    stream_count = task_assignment(
      levels,
      dfg,
      tasks,
      max_stream_count
    );
    std::cout << "max_stream_count " << max_stream_count << std::endl;
    std::cout << "stream_count " << stream_count << std::endl;

  }

  // check_nodes(dfg);

  // 

  constexpr int N = 1 << 20;

  float* dmem = nullptr;

  CUDA_CHECK(cudaMalloc(&dmem, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(dmem, 0, N * sizeof(float)));


  RuntimeResources rr;

  {
    ScopedTimer timer(bench.resource_allocation_ms);
    // rr = resource_allocation(dfg, stream_count);
    rr = resource_allocation(dfg, tasks, stream_count);
  }

  tf::Executor executor(8);
  tf::Taskflow tf;

  std::unordered_map<int, tf::Task> task_nodes;
  std::unordered_map<int, cudaEvent_t> done_events;
  std::unordered_map<int, cudaEvent_t> kernel_start_events;
  std::unordered_map<int, cudaEvent_t> kernel_stop_events;

  // 全ストリーム上のイベント時刻を比較するための共通基準時刻。
  cudaStream_t timing_stream = nullptr;
  cudaEvent_t origin_event = nullptr;

  CUDA_CHECK(cudaStreamCreateWithFlags(
    &timing_stream,
    cudaStreamNonBlocking
  ));
  CUDA_CHECK(cudaEventCreate(&origin_event));
  CUDA_CHECK(cudaEventRecord(origin_event, timing_stream));
  CUDA_CHECK(cudaEventSynchronize(origin_event));

  for (const auto& t : tasks) {
    cudaEvent_t done_ev;
    cudaEvent_t start_ev;
    cudaEvent_t stop_ev;

    CUDA_CHECK(cudaEventCreateWithFlags(
      &done_ev,
      cudaEventDisableTiming
    ));
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    done_events.emplace(t.id, done_ev);
    kernel_start_events.emplace(t.id, start_ev);
    kernel_stop_events.emplace(t.id, stop_ev);
  }

  for (const auto& t : tasks) {
    tf::Task node = tf.emplace([&, t]() {
      const NodeInfo& info = dfg.nodes.at(t.id);
      cudaStream_t stream = get_stream_by_id(rr, info.stream_id);

      for (int pred : t.preds) {
        auto it = done_events.find(pred);

        if (it == done_events.end()) {
          throw std::runtime_error(
            "unknown predecessor id: " + std::to_string(pred)
          );
        }

        CUDA_CHECK(cudaStreamWaitEvent(stream, it->second, 0));
      }

      // start/stop はGPU上のカーネル実行区間だけを測る。
      CUDA_CHECK(cudaEventRecord(kernel_start_events.at(t.id), stream));
      launch_task_kernel(t, dmem, N, stream);
      CUDA_CHECK(cudaEventRecord(kernel_stop_events.at(t.id), stream));

      // 後続タスクは、このカーネルの完了を待つ。
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

  executor.run(tf).wait();

  // Taskflow の完了はCUDA処理の投入完了であり、GPU完了ではない。
  CUDA_CHECK(cudaDeviceSynchronize());

  // GPU_SUBMIT_WAIT_MS:
  // 並列実行時の「最初のカーネル開始」から
  // 「最後のカーネル終了」までのGPUタイムライン上の時間。
  // CPU側のTaskflow処理や、最初のカーネル投入前の時間は含めない。
  if (tasks.empty()) {
    bench.gpu_submit_wait_ms = 0.0;
  } else {
    double first_kernel_start_ms =
      std::numeric_limits<double>::max();
    double last_kernel_stop_ms = 0.0;

    for (const auto& t : tasks) {
      float start_from_origin_ms = 0.0f;
      float stop_from_origin_ms = 0.0f;

      CUDA_CHECK(cudaEventElapsedTime(
        &start_from_origin_ms,
        origin_event,
        kernel_start_events.at(t.id)
      ));

      CUDA_CHECK(cudaEventElapsedTime(
        &stop_from_origin_ms,
        origin_event,
        kernel_stop_events.at(t.id)
      ));

      first_kernel_start_ms = std::min(
        first_kernel_start_ms,
        static_cast<double>(start_from_origin_ms)
      );

      last_kernel_stop_ms = std::max(
        last_kernel_stop_ms,
        static_cast<double>(stop_from_origin_ms)
      );
    }

    bench.gpu_submit_wait_ms =
      last_kernel_stop_ms - first_kernel_start_ms;
  }
    
  print_runtime_resource_state(rr, stream_count, "before GC release");

  release_finished_gc_streams(rr, stream_count);

  print_runtime_resource_state(rr, stream_count, "after GC release");

  if (all_gc_streams_released(rr, stream_count)) {
    std::cout << "[CHECK] all GC streams released\n";
  } else {
    std::cout << "[CHECK] some GC streams are still alive\n";
  }

  for (auto& [id, ev] : done_events) {
    CUDA_CHECK(cudaEventDestroy(ev));
  }

  for (auto& [id, ev] : kernel_start_events) {
    CUDA_CHECK(cudaEventDestroy(ev));
  }

  for (auto& [id, ev] : kernel_stop_events) {
    CUDA_CHECK(cudaEventDestroy(ev));
  }

  CUDA_CHECK(cudaEventDestroy(origin_event));
  CUDA_CHECK(cudaStreamDestroy(timing_stream));

  destroy_runtime_resources(rr);

  CUDA_CHECK(cudaFree(dmem));

  auto total_end = Clock::now();

  bench.total_ms = elapsed_ms(total_start, total_end);

  // 並列実行本体とは別に、全カーネルを単一ストリームで実測する。
  // この追加ベンチマーク時間は TOTAL_MS には含めない。
  bench.gpu_kernel_ms = measure_sequential_kernel_ms(tasks, N);

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