

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

// mkdir -p "$HOME/tmp/nsys"
// TMPDIR="$HOME/tmp/nsys" nsys profile \
//   --force-overwrite true \
//   --trace=cuda,nvtx,osrt \
//   --gpu-metrics-devices=0 \
//   --gpu-metrics-frequency=10000 \
//   -o stg_profile \
//   ./main

// If /tmp is not writable, set a user-owned temp directory explicitly:
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
#include "../stg_common.hpp"
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

// 1タスクは最大64 SM分のblockを持つ。
// 単体実行では114 SMを使い切らず、独立タスクの並列実行で
// 空いているSMを利用できるベンチマークにする。
constexpr int kTaskParallelSmLimit = 64;
constexpr int kTaskElementCount =
    kTaskParallelSmLimit * 256;



StgGraph load_stg_without_comm(const std::string& path) {
  return stg::load_stg_without_comm(path);
}

static int safe_work_units(int proc_time, int scale, int minimum) {
  return stg::safe_work_units(proc_time, scale, minimum);
}

std::vector<TaskSpec> make_task_specs_from_stg(const StgGraph& g) {
  return stg::make_task_specs_from_stg_common<TaskSpec>(
      g,
      [](TaskSpec& s, const stg::StgTask& t, int threshold) {
        s.parallel_sm_limit = kTaskParallelSmLimit;

        if (t.proc_time > threshold) {
          s.kind = KernelKind::HEAVY;
          s.work_units = stg::safe_work_units(t.proc_time, 200, 1000);
        } else {
          s.kind = KernelKind::LIGHT;
          s.work_units = stg::safe_work_units(t.proc_time, 80, 200);
        }
      });
}

void launch_task_kernel(const TaskSpec& task,
                        float* dmem,
                        int n,
                        cudaStream_t stream) {
  stg::launch_task_kernel(task, dmem, n, stream);
}

/*
 * 比較用の逐次実行。
 * Green Contextを作らず、primary context上の1 streamへ
 * レベル順に全kernelを投入し、GPU上のkernel時間だけを測る。
 */
double run_sequential_cuda(
    const std::vector<TaskSpec>& tasks
) {
  TaskDFG dfg = task_DFG_construction(tasks);
  const TaskLevels levels = task_levelization(dfg);
  const auto task_by_id = make_task_table(tasks);

  constexpr int N = kTaskElementCount;
  float* dmem = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;

  CUDA_CHECK(
      cudaMalloc(
          &dmem,
          static_cast<std::size_t>(N) * sizeof(float)
      )
  );

  CUDA_CHECK(
      cudaMemset(
          dmem,
          0,
          static_cast<std::size_t>(N) * sizeof(float)
      )
  );

  CUDA_CHECK(
      cudaStreamCreateWithFlags(
          &stream,
          cudaStreamNonBlocking
      )
  );

  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));
  CUDA_CHECK(cudaEventRecord(start_event, stream));

  for (const auto& level : levels) {
    for (const int task_id : level) {
      launch_task_kernel(
          *task_by_id.at(task_id),
          dmem,
          N,
          stream
      );
    }
  }

  CUDA_CHECK(cudaEventRecord(stop_event, stream));
  CUDA_CHECK(cudaEventSynchronize(stop_event));

  float sequential_gpu_ms = 0.0f;

  CUDA_CHECK(
      cudaEventElapsedTime(
          &sequential_gpu_ms,
          start_event,
          stop_event
      )
  );

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dmem));

  return static_cast<double>(sequential_gpu_ms);
}

// 実行
BenchResult run_taskflow_cuda(
    const std::vector<TaskSpec>& tasks
) {
  BenchResult bench;
  const auto total_start = Clock::now();

  // 1. DFG構築
  TaskDFG dfg;
  {
    ScopedTimer timer(bench.task_DFG_construction_ms);
    dfg = task_DFG_construction(tasks);
  }

  // 2. レベル化
  TaskLevels levels;
  {
    ScopedTimer timer(bench.task_levelization_ms);
    levels = task_levelization(dfg);
  }

  constexpr int max_stream_count = 5;
  constexpr int scheduling_reference_sm = 114;

  /*
   * 3. STG_existing_methodと同じ規則でstream数を決める（最大5本）。
   * この段階ではGCのSM配分を行わない。
   */
  const int stream_count =
      decide_stream_count(levels, max_stream_count);

  /*
   * 4. 既存手法と同じ規則でタスクをstreamへ割り当てる。
   * この時点ではGreen ContextのSM配分を決めない。
   * 全streamを同じ性能として扱うため、GCの配分結果は
   * タスク割り当てに影響しない。
   */
  const std::vector<int> scheduling_stream_sm_counts(
      static_cast<std::size_t>(stream_count),
      scheduling_reference_sm
  );

  StreamScheduleResult initial_schedule;

  {
    ScopedTimer timer(bench.task_assignment_ms);

    initial_schedule =
        simulate_stream_assignment(
            tasks,
            scheduling_stream_sm_counts,
            scheduling_reference_sm
        );
  }

  /*
   * 5. タスクのstream割り当て確定後に、GCのSMを均等配分する。
   * この配分を使ったタスクの再割り当ては行わない。
   */
  const std::vector<int> stream_sm_counts =
      make_even_gc_sm_counts(stream_count);

  const StreamScheduleResult final_schedule =
      initial_schedule;

  apply_stream_schedule_to_dfg(
      dfg,
      final_schedule
  );

  if (static_cast<int>(stream_sm_counts.size()) != stream_count) {
    throw std::runtime_error(
        "stream_sm_counts size does not match stream_count"
    );
  }

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    if (stream_sm_counts.at(
            static_cast<std::size_t>(stream_id)
        ) <= 0) {
      throw std::runtime_error(
          "invalid SM count for stream " +
          std::to_string(stream_id)
      );
    }
  }

  std::cout << "===== Existing method + even GC configuration =====\n";
  std::cout << "max_stream_count : "
            << max_stream_count << '\n';
  std::cout << "stream_count     : "
            << stream_count << '\n';
  std::cout << "estimated makespan: "
            << final_schedule.makespan
            << '\n';

  double sequential_estimated_time = 0.0;

  for (const auto& task : tasks) {
    sequential_estimated_time +=
        static_cast<double>(
            std::max(0, task.proc_time)
        );
  }

  const double estimated_speedup =
      final_schedule.makespan <= 0.0
          ? 0.0
          : sequential_estimated_time /
                final_schedule.makespan;

  const double estimated_reduction_percent =
      sequential_estimated_time <= 0.0
          ? 0.0
          : (sequential_estimated_time -
             final_schedule.makespan) /
                sequential_estimated_time * 100.0;

  std::cout << "sequential estimated: "
            << sequential_estimated_time
            << '\n';
  std::cout << "estimated speedup    : "
            << estimated_speedup
            << " x\n";
  std::cout << "estimated reduction  : "
            << estimated_reduction_percent
            << " %\n";

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    std::cout
        << "stream "
        << stream_id
        << " : "
        << stream_sm_counts.at(
               static_cast<std::size_t>(stream_id)
           )
        << " SM\n";
  }

  std::vector<int> planned_task_counts(
      static_cast<std::size_t>(stream_count),
      0
  );
  std::vector<long long> planned_proc_loads(
      static_cast<std::size_t>(stream_count),
      0
  );

  for (const auto& task : tasks) {
    const int stream_id =
        final_schedule.task_stream.at(task.id);

    ++planned_task_counts.at(
        static_cast<std::size_t>(stream_id)
    );
    planned_proc_loads.at(
        static_cast<std::size_t>(stream_id)
    ) += std::max(0, task.proc_time);
  }

  std::cout << "===== Planned stream load =====\n";

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    std::cout
        << "stream " << stream_id
        << " : tasks="
        << planned_task_counts.at(
               static_cast<std::size_t>(stream_id)
           )
        << ", proc_load="
        << planned_proc_loads.at(
               static_cast<std::size_t>(stream_id)
           )
        << '\n';
  }

  std::cout << "===============================\n";

  /*
   * stream 0がどれだけ密に埋まったかを表示する。
   * density = busy time / (last finish - first start)
   */
  double normal_busy_time = 0.0;
  double normal_first_start =
      std::numeric_limits<double>::max();
  double normal_last_finish = 0.0;
  int normal_task_count = 0;

  for (const auto& task : tasks) {
    if (final_schedule.task_stream.at(task.id) != 0) {
      continue;
    }

    const double start =
        final_schedule.task_start_time.at(task.id);
    const double finish =
        final_schedule.task_finish_time.at(task.id);

    normal_busy_time += finish - start;
    normal_first_start = std::min(normal_first_start, start);
    normal_last_finish = std::max(normal_last_finish, finish);
    ++normal_task_count;
  }

  const double normal_span =
      normal_task_count == 0
          ? 0.0
          : normal_last_finish - normal_first_start;

  const double normal_density =
      normal_span <= 0.0
          ? 0.0
          : normal_busy_time / normal_span;

  std::cout << "stream 0 tasks       : "
            << normal_task_count << '\n';
  std::cout << "stream 0 busy        : "
            << normal_busy_time << '\n';
  std::cout << "stream 0 span        : "
            << normal_span << '\n';
  std::cout << "stream 0 density     : "
            << normal_density << '\n';

  constexpr int N = kTaskElementCount;

  float* dmem = nullptr;

  CUDA_CHECK(
      cudaMalloc(
          &dmem,
          static_cast<std::size_t>(N) *
              static_cast<std::size_t>(stream_count) *
              sizeof(float)
      )
  );

  CUDA_CHECK(
      cudaMemset(
          dmem,
          0,
          static_cast<std::size_t>(N) *
              static_cast<std::size_t>(stream_count) *
              sizeof(float)
      )
  );

  /*
   * 6. 決定済みのSM数を使用して実行資源を生成
   */
  RuntimeResources rr;

  {
    ScopedTimer timer(bench.resource_allocation_ms);

    rr = create_runtime_resources(
        stream_sm_counts
    );
  }

  tf::Executor executor(8);
  tf::Taskflow taskflow;

  std::unordered_map<int, tf::Task> task_nodes;
  std::unordered_map<int, cudaEvent_t> done_events;
  std::unordered_map<int, cudaEvent_t> kernel_start_events;
  std::unordered_map<int, cudaEvent_t> kernel_stop_events;

  task_nodes.reserve(tasks.size());
  done_events.reserve(tasks.size());
  kernel_start_events.reserve(tasks.size());
  kernel_stop_events.reserve(tasks.size());

  /*
   * 全streamのイベント時刻を比較するための共通基準
   */
  cudaStream_t timing_stream = nullptr;
  cudaEvent_t origin_event = nullptr;

  CUDA_CHECK(
      cudaStreamCreateWithFlags(
          &timing_stream,
          cudaStreamNonBlocking
      )
  );

  CUDA_CHECK(
      cudaEventCreate(&origin_event)
  );

  CUDA_CHECK(
      cudaEventRecord(
          origin_event,
          timing_stream
      )
  );

  CUDA_CHECK(
      cudaEventSynchronize(origin_event)
  );

  /*
   * タスクごとのCUDA Eventを生成
   */
  for (const auto& task : tasks) {
    cudaEvent_t done_event = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    CUDA_CHECK(
        cudaEventCreateWithFlags(
            &done_event,
            cudaEventDisableTiming
        )
    );

    CUDA_CHECK(
        cudaEventCreate(&start_event)
    );

    CUDA_CHECK(
        cudaEventCreate(&stop_event)
    );

    done_events.emplace(
        task.id,
        done_event
    );

    kernel_start_events.emplace(
        task.id,
        start_event
    );

    kernel_stop_events.emplace(
        task.id,
        stop_event
    );
  }

  /*
   * Taskflowノードを生成
   */
  for (const auto& task : tasks) {
    tf::Task node =
        taskflow.emplace([&, task]() {
          const auto node_it =
              dfg.nodes.find(task.id);

          if (node_it == dfg.nodes.end()) {
            throw std::runtime_error(
                "DFG node not found: " +
                std::to_string(task.id)
            );
          }

          const int stream_id =
              node_it->second.stream_id;

          if (stream_id < 0 ||
              stream_id >= stream_count) {
            throw std::runtime_error(
                "invalid stream id for task " +
                std::to_string(task.id)
            );
          }

          cudaStream_t stream =
              get_stream_by_id(
                  rr,
                  stream_id
              );

          /*
           * 先行タスクのGPU完了を待つ
           */
          for (const int pred_id : task.preds) {
            const auto event_it =
                done_events.find(pred_id);

            if (event_it == done_events.end()) {
              throw std::runtime_error(
                  "unknown predecessor id: " +
                  std::to_string(pred_id)
              );
            }

            CUDA_CHECK(
                cudaStreamWaitEvent(
                    stream,
                    event_it->second,
                    0
                )
            );
          }

          CUDA_CHECK(
              cudaEventRecord(
                  kernel_start_events.at(task.id),
                  stream
              )
          );

          launch_task_kernel(
              task,
              dmem +
                  static_cast<std::size_t>(stream_id) *
                      static_cast<std::size_t>(N),
              N,
              stream
          );

          CUDA_CHECK(
              cudaEventRecord(
                  kernel_stop_events.at(task.id),
                  stream
              )
          );

          CUDA_CHECK(
              cudaEventRecord(
                  done_events.at(task.id),
                  stream
              )
          );
        });

    node.name(
        std::to_string(task.id)
    );

    task_nodes.emplace(
        task.id,
        node
    );
  }

  /*
   * Taskflow上の依存関係を設定
   */
  for (const auto& task : tasks) {
    const auto current_it =
        task_nodes.find(task.id);

    if (current_it == task_nodes.end()) {
      throw std::runtime_error(
          "Taskflow node not found: " +
          std::to_string(task.id)
      );
    }

    for (const int pred_id : task.preds) {
      const auto predecessor_it =
          task_nodes.find(pred_id);

      if (predecessor_it == task_nodes.end()) {
        throw std::runtime_error(
            "predecessor Taskflow node not found: " +
            std::to_string(pred_id)
        );
      }

      predecessor_it->second.precede(
          current_it->second
      );
    }
  }

  /*
   * 計画した開始時刻順に、同一stream内の投入順を固定する。
   * 依存待ちタスクが先にstreamへ投入されて後続を塞ぐ
   * Head-of-Line Blockingを防ぐ。
   */
  std::vector<std::vector<int>> stream_task_order(
      static_cast<std::size_t>(stream_count)
  );

  std::unordered_map<int, const TaskSpec*> task_spec_by_id;
  task_spec_by_id.reserve(tasks.size());

  for (const auto& task : tasks) {
    task_spec_by_id.emplace(task.id, &task);
    const int stream_id =
        final_schedule.task_stream.at(task.id);

    stream_task_order.at(
        static_cast<std::size_t>(stream_id)
    ).push_back(task.id);
  }

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    auto& task_ids = stream_task_order.at(
        static_cast<std::size_t>(stream_id)
    );

    std::stable_sort(
        task_ids.begin(),
        task_ids.end(),
        [&](int lhs_id, int rhs_id) {
          const double lhs_start =
              final_schedule.task_start_time.at(lhs_id);
          const double rhs_start =
              final_schedule.task_start_time.at(rhs_id);

          if (lhs_start != rhs_start) {
            return lhs_start < rhs_start;
          }

          const double lhs_finish =
              final_schedule.task_finish_time.at(lhs_id);
          const double rhs_finish =
              final_schedule.task_finish_time.at(rhs_id);

          if (lhs_finish != rhs_finish) {
            return lhs_finish < rhs_finish;
          }

          return lhs_id < rhs_id;
        }
    );

    for (std::size_t index = 1;
         index < task_ids.size();
         ++index) {
      const int previous_id = task_ids.at(index - 1);
      const int current_id = task_ids.at(index);

      bool already_direct_predecessor = false;

      const auto current_spec_it =
          task_spec_by_id.find(current_id);

      if (current_spec_it == task_spec_by_id.end()) {
        throw std::runtime_error(
            "task specification not found: " +
            std::to_string(current_id)
        );
      }

      for (const int pred_id : current_spec_it->second->preds) {
        if (pred_id == previous_id) {
          already_direct_predecessor = true;
          break;
        }
      }

      if (!already_direct_predecessor) {
        task_nodes.at(previous_id).precede(
            task_nodes.at(current_id)
        );
      }
    }
  }

  /*
   * trial/baselineと同じ定義で計測する。
   *
   * gpu_submit_wait_ms:
   *   CPU側でTaskflowの投入を開始してから、GPU上の全処理が
   *   完了するまでの実時間。
   *
   * gpu_kernel_ms:
   *   GPU上で最初のkernelが開始してから、最後のkernelが
   *   終了するまでのCUDA Event時間。
   */
  const auto submit_start = Clock::now();

  executor.run(taskflow).wait();

  /*
   * Taskflow完了はCUDAコマンドの投入完了なので、
   * GPU上の全kernel完了を明示的に待つ。
   */
  CUDA_CHECK(
      cudaDeviceSynchronize()
  );

  const auto submit_end = Clock::now();

  bench.gpu_submit_wait_ms =
      elapsed_ms(
          submit_start,
          submit_end
      );

  /*
   * 複数stream全体のGPU makespanを求める。
   * 各taskの開始・終了イベントを共通originから測り、
   * 最初の開始から最後の終了までをgpu_kernel_msとする。
   */
  if (tasks.empty()) {
    bench.gpu_kernel_ms = 0.0;
  } else {
    double first_kernel_start_ms =
        std::numeric_limits<double>::max();

    double last_kernel_stop_ms = 0.0;

    for (const auto& task : tasks) {
      float start_from_origin_ms = 0.0f;
      float stop_from_origin_ms = 0.0f;

      CUDA_CHECK(
          cudaEventElapsedTime(
              &start_from_origin_ms,
              origin_event,
              kernel_start_events.at(task.id)
          )
      );

      CUDA_CHECK(
          cudaEventElapsedTime(
              &stop_from_origin_ms,
              origin_event,
              kernel_stop_events.at(task.id)
          )
      );

      first_kernel_start_ms =
          std::min(
              first_kernel_start_ms,
              static_cast<double>(
                  start_from_origin_ms
              )
          );

      last_kernel_stop_ms =
          std::max(
              last_kernel_stop_ms,
              static_cast<double>(
                  stop_from_origin_ms
              )
          );
    }

    bench.gpu_kernel_ms =
        last_kernel_stop_ms -
        first_kernel_start_ms;
  }

  print_runtime_resource_state(
      rr,
      stream_count,
      "before GC release"
  );

  release_finished_gc_streams(
      rr,
      stream_count
  );

  print_runtime_resource_state(
      rr,
      stream_count,
      "after GC release"
  );

  if (all_gc_streams_released(
          rr,
          stream_count
      )) {
    std::cout
        << "[CHECK] all GC streams released\n";
  } else {
    std::cout
        << "[CHECK] some GC streams are still alive\n";
  }

  /*
   * CUDA Eventを破棄
   */
  for (auto& [task_id, event] : done_events) {
    CUDA_CHECK(
        cudaEventDestroy(event)
    );
  }

  for (auto& [task_id, event] : kernel_start_events) {
    CUDA_CHECK(
        cudaEventDestroy(event)
    );
  }

  for (auto& [task_id, event] : kernel_stop_events) {
    CUDA_CHECK(
        cudaEventDestroy(event)
    );
  }

  CUDA_CHECK(
      cudaEventDestroy(origin_event)
  );

  CUDA_CHECK(
      cudaStreamDestroy(timing_stream)
  );

  destroy_runtime_resources(rr);

  CUDA_CHECK(
      cudaFree(dmem)
  );

  const auto total_end = Clock::now();

  bench.total_ms =
      elapsed_ms(
          total_start,
          total_end
      );

  return bench;
}

void print_stg_summary(const std::vector<TaskSpec>& tasks) {
  stg::print_stg_summary(tasks);
  std::cout << "task_grid     : " << kTaskParallelSmLimit << " blocks\n";
  std::cout << "task_SM_limit : " << kTaskParallelSmLimit << " SM\n";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr
        << "Usage: "
        << argv[0]
        << " input.stg\n";

    return 1;
  }

  try {
    const StgGraph graph =
        load_stg_without_comm(argv[1]);

    const auto tasks =
        make_task_specs_from_stg(graph);

    print_stg_summary(tasks);

    const double sequential_gpu_kernel_ms =
        run_sequential_cuda(tasks);

    const BenchResult result =
        run_taskflow_cuda(tasks);

    const double measured_speedup =
        result.gpu_kernel_ms <= 0.0
            ? 0.0
            : sequential_gpu_kernel_ms /
                  result.gpu_kernel_ms;

    const double measured_reduction_percent =
        sequential_gpu_kernel_ms <= 0.0
            ? 0.0
            : (sequential_gpu_kernel_ms -
               result.gpu_kernel_ms) /
                  sequential_gpu_kernel_ms * 100.0;

    std::cout << "===== Sequential comparison =====\n";
    std::cout << "sequential gpu_kernel_ms: "
              << sequential_gpu_kernel_ms
              << " ms\n";
    std::cout << "existing+GC gpu_kernel_ms: "
              << result.gpu_kernel_ms
              << " ms\n";
    std::cout << "measured speedup          : "
              << measured_speedup
              << " x\n";
    std::cout << "measured reduction        : "
              << measured_reduction_percent
              << " %\n";
    std::cout << "=================================\n";

    std::cout << "mode: existing_method_GC\n";
    print_result(result);
  }
  catch (const std::exception& error) {
    std::cerr
        << "exception: "
        << error.what()
        << '\n';

    return 1;
  }

  return 0;
}
