

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
// mkdir -p "$HOME/tmp/nsys"

// TMPDIR="$HOME/tmp/nsys" nsys profile \
//   --force-overwrite=true \
//   --trace=cuda,nvtx,osrt \
//   --gpu-metrics-devices=0 \
//   --gpu-metrics-frequency=10000 \
//   -o stg_profile_proposed \
//   ./main sample.stg proposed

#include <cuda_runtime.h>

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
#include <thread>
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

  /*
   * 旧baselineと同じ計算量。
   */
  s.work_units = std::max(
      1000,
      t.proc_time * 200
  );
}
else {
  s.kind = KernelKind::LIGHT;

  /*
   * 旧baselineと同じ計算量。
   * proc_time=0のダミータスクも200反復で起動する。
   */
  s.work_units = std::max(
      200,
      std::max(1, t.proc_time) * 80
  );
}

    specs.push_back(std::move(s));
  }

  return specs;
}

__global__ void light_kernel(float* data,
                               int n,
                               int iters) {
  const int idx =
      blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float x = data[idx];

    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
      x = x * 1.000001f + 0.00001f;
    }

    data[idx] = x;
  }
}

__global__ void heavy_kernel(float* data,
                               int n,
                               int iters) {
  const int idx =
      blockIdx.x * blockDim.x + threadIdx.x;

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
  if (task.work_units <= 0) {
    return;
  }

  constexpr int threads_per_block = 256;

  const dim3 block(threads_per_block);

  /*
   * baseline/proposedで完全に同じ起動条件。
   * N=1<<20、block=256なので4096 blocks。
   */
  const dim3 grid(
      (n + threads_per_block - 1) /
      threads_per_block
  );

  if (task.kind == KernelKind::HEAVY) {
    heavy_kernel<<<grid, block, 0, stream>>>(
        dmem,
        n,
        task.work_units
    );
  }
  else {
    light_kernel<<<grid, block, 0, stream>>>(
        dmem,
        n,
        task.work_units
    );
  }

  CUDA_CHECK(cudaGetLastError());
}


/*
 * CPU側でCUDAコマンドを投入する順序を作る。
 *
 * 重要:
 *   - predecessorは必ずsuccessorより先に投入する。
 *   - 同時に投入可能なtaskの中では、
 *     HEFTの計画開始時刻が早いtaskを優先する。
 *
 * CPUはGPU完了を待たない。
 * 依存関係はcudaStreamWaitEventでGPU側に任せる。
 */
std::vector<const TaskSpec*> make_cuda_enqueue_order(
    const std::vector<TaskSpec>& tasks,
    const StreamScheduleResult& schedule
) {
  std::unordered_map<int, const TaskSpec*> task_by_id;
  std::unordered_map<int, int> remaining_predecessors;
  std::unordered_map<int, std::vector<int>> successors;

  task_by_id.reserve(tasks.size());
  remaining_predecessors.reserve(tasks.size());
  successors.reserve(tasks.size());

  for (const auto& task : tasks) {
    const auto [it, inserted] =
        task_by_id.emplace(task.id, &task);

    if (!inserted) {
      throw std::runtime_error(
          "duplicate task id: " +
          std::to_string(task.id)
      );
    }

    remaining_predecessors.emplace(
        task.id,
        static_cast<int>(task.preds.size())
    );

    successors.emplace(
        task.id,
        std::vector<int>{}
    );
  }

  for (const auto& task : tasks) {
    for (const int pred_id : task.preds) {
      const auto pred_it =
          successors.find(pred_id);

      if (pred_it == successors.end()) {
        throw std::runtime_error(
            "unknown predecessor id: " +
            std::to_string(pred_id)
        );
      }

      pred_it->second.push_back(task.id);
    }
  }

  std::vector<int> ready;
  ready.reserve(tasks.size());

  for (const auto& task : tasks) {
    if (remaining_predecessors.at(task.id) == 0) {
      ready.push_back(task.id);
    }
  }

  const auto comes_before =
      [&](int lhs_id, int rhs_id) {
        const double lhs_start =
            schedule.task_start_time.at(lhs_id);

        const double rhs_start =
            schedule.task_start_time.at(rhs_id);

        if (lhs_start != rhs_start) {
          return lhs_start < rhs_start;
        }

        const double lhs_finish =
            schedule.task_finish_time.at(lhs_id);

        const double rhs_finish =
            schedule.task_finish_time.at(rhs_id);

        if (lhs_finish != rhs_finish) {
          return lhs_finish < rhs_finish;
        }

        return lhs_id < rhs_id;
      };

  std::vector<const TaskSpec*> order;
  order.reserve(tasks.size());

  while (!ready.empty()) {
    const auto next_it =
        std::min_element(
            ready.begin(),
            ready.end(),
            comes_before
        );

    const int task_id = *next_it;
    ready.erase(next_it);

    order.push_back(
        task_by_id.at(task_id)
    );

    for (const int successor_id :
         successors.at(task_id)) {
      int& predecessor_count =
          remaining_predecessors.at(successor_id);

      --predecessor_count;

      if (predecessor_count < 0) {
        throw std::runtime_error(
            "invalid predecessor count for task " +
            std::to_string(successor_id)
        );
      }

      if (predecessor_count == 0) {
        ready.push_back(successor_id);
      }
    }
  }

  if (order.size() != tasks.size()) {
    throw std::runtime_error(
        "cycle detected while building CUDA enqueue order"
    );
  }

  return order;
}

/*
 * 1 stream内のkernel区間を結合する。
 */
std::vector<std::pair<double, double>>
merge_intervals(
    std::vector<std::pair<double, double>> intervals
) {
  if (intervals.empty()) {
    return {};
  }

  std::sort(
      intervals.begin(),
      intervals.end()
  );

  std::vector<std::pair<double, double>> merged;

  for (const auto& interval : intervals) {
    if (interval.second < interval.first) {
      throw std::runtime_error(
          "invalid measured kernel interval"
      );
    }

    if (merged.empty() ||
        interval.first > merged.back().second) {
      merged.push_back(interval);
    }
    else {
      merged.back().second =
          std::max(
              merged.back().second,
              interval.second
          );
    }
  }

  return merged;
}

/*
 * 2 streamのkernel実行区間が実際に重なった時間を返す。
 */
double calculate_interval_overlap_ms(
    const std::vector<std::pair<double, double>>& lhs,
    const std::vector<std::pair<double, double>>& rhs
) {
  std::size_t lhs_index = 0;
  std::size_t rhs_index = 0;
  double overlap_ms = 0.0;

  while (lhs_index < lhs.size() &&
         rhs_index < rhs.size()) {
    const double overlap_start =
        std::max(
            lhs.at(lhs_index).first,
            rhs.at(rhs_index).first
        );

    const double overlap_stop =
        std::min(
            lhs.at(lhs_index).second,
            rhs.at(rhs_index).second
        );

    if (overlap_stop > overlap_start) {
      overlap_ms +=
          overlap_stop - overlap_start;
    }

    if (lhs.at(lhs_index).second <
        rhs.at(rhs_index).second) {
      ++lhs_index;
    }
    else {
      ++rhs_index;
    }
  }

  return overlap_ms;
}

// 実行
BenchResult run_cuda_schedule(
    const std::vector<TaskSpec>& tasks,
    const std::string& mode
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

  /*
   * 3. proc_timeを使ってStream本数、SM配分、タスク配置を決める。
   *
   * 各候補本数について以下を反復する。
   *   1. DFGのready taskをbottom level順に選ぶ。
   *   2. proc_timeから予測終了時刻を計算する。
   *   3. 最も早く終了するStreamへ配置する。
   *   4. 各Streamへ配置されたproc_time総量に比例してSMを再配分する。
   *
   * 1本から最大5本まで比較し、予測makespanが最小の構成を採用する。
   */
  StreamConfigurationDecision stream_decision;
  StreamScheduleResult selected_schedule;
  ProcTimeSmModel scheduling_model;

  double normal_only_makespan = 0.0;
  int evaluated_configuration_count = 0;

  {
    ScopedTimer timer(bench.resource_allocation_ms);

    const SmPartitionInfo partition_info =
        query_sm_partition_info();

    const ProcTimeConfigurationResult
        proc_time_configuration =
            decide_proc_time_stream_configuration(
                tasks,
                levels,
                max_stream_count,
                partition_info.available_sm,
                partition_info.unit_sm,
                partition_info.min_group_sm
            );

    scheduling_model = proc_time_configuration.model;
    normal_only_makespan =
        proc_time_configuration.normal_only_makespan;

    if (mode == "baseline") {
      const ProcTimeCandidateSummary& normal_candidate =
          proc_time_configuration.candidates.front();

      stream_decision.stream_count = 1;
      stream_decision.stream_sm_counts =
          normal_candidate.stream_sm_counts;
      stream_decision.estimated_makespan =
          normal_candidate.estimated_makespan;

      selected_schedule =
          simulate_proc_time_assignment(
              tasks,
              stream_decision.stream_sm_counts,
              scheduling_model
          );

      evaluated_configuration_count = 1;
    }
    else {
      stream_decision =
          proc_time_configuration.decision;
      selected_schedule =
          proc_time_configuration.schedule;
      evaluated_configuration_count =
          proc_time_configuration.evaluated_configuration_count;
    }
  }

  const int stream_count =
      stream_decision.stream_count;

  const std::vector<int> stream_sm_counts =
      stream_decision.stream_sm_counts;

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

  /*
   * 4. proc_timeで決定済みのスケジュールをDFGへ反映する。
   */
  StreamScheduleResult final_schedule =
      selected_schedule;

  {
    ScopedTimer timer(bench.task_assignment_ms);

    apply_stream_schedule_to_dfg(
        dfg,
        final_schedule
    );
  }

  std::cout
      << "===== Stream selection =====\n";

  std::cout
      << "stream_count              : "
      << stream_count
      << '\n';

  std::cout
      << "SM allocation             : ";

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    if (stream_id > 0) {
      std::cout << ',';
    }

    std::cout
        << stream_sm_counts.at(
               static_cast<std::size_t>(stream_id)
           );
  }

  std::cout << '\n';

  std::cout
      << "normal-only estimate      : "
      << normal_only_makespan
      << '\n';

  std::cout
      << "selected estimate         : "
      << stream_decision.estimated_makespan
      << '\n';

  std::cout
      << "evaluated configurations  : "
      << evaluated_configuration_count
      << '\n';

  std::cout
      << "decision metric           : "
      << "proc_time based EFT"
      << '\n';

  std::cout
      << "task SM demand range      : "
      << scheduling_model.min_task_sm
      << '-'
      << scheduling_model.max_task_sm
      << '\n';

  if (mode == "baseline") {
    std::cout
        << "reason                    : "
        << "comparison baseline; one stream is forced"
        << '\n';
  }
  else if (stream_count == 1) {
    std::cout
        << "reason                    : "
        << "normal-only had the minimum estimated makespan"
        << '\n';
  }
  else {
    const double reduction_percent =
        normal_only_makespan > 0.0
            ? (
                  normal_only_makespan -
                  stream_decision.estimated_makespan
              ) /
                  normal_only_makespan *
                  100.0
            : 0.0;

    std::cout
        << "reason                    : "
        << stream_count
        << " streams had the minimum estimate; "
        << reduction_percent
        << "% smaller than normal-only"
        << '\n';
  }

  std::cout
      << "============================\n";


constexpr int N = 1 << 20;

/*
 * streamごとに独立した作業領域を使用する。
 * 異なるstream間の同時書込み競合を防ぐ。
 */
const std::size_t total_element_count =
    static_cast<std::size_t>(stream_count) *
    static_cast<std::size_t>(N);

float* dmem = nullptr;

CUDA_CHECK(
    cudaMalloc(
        &dmem,
        total_element_count * sizeof(float)
    )
);

CUDA_CHECK(
    cudaMemset(
        dmem,
        0,
        total_element_count * sizeof(float)
    )
);

  /*
   * 6. 決定済みのSM数を使用してGreen Contextを生成
   */
  RuntimeResources rr;

  {
    ScopedTimer timer(bench.resource_allocation_ms);

    rr = create_runtime_resources(
        stream_sm_counts
    );
  }

  std::unordered_map<int, cudaEvent_t> done_events;
  std::unordered_map<int, cudaEvent_t> kernel_start_events;
  std::unordered_map<int, cudaEvent_t> kernel_stop_events;

  done_events.reserve(tasks.size());
  kernel_start_events.reserve(tasks.size());
  kernel_stop_events.reserve(tasks.size());

  /*
   * 全streamのevent時刻を比較するための共通基準。
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
   * taskごとのCUDA eventを作る。
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

  const std::vector<const TaskSpec*> enqueue_order =
      make_cuda_enqueue_order(
          tasks,
          final_schedule
      );

  std::vector<int> task_count_per_stream(
      static_cast<std::size_t>(stream_count),
      0
  );

  for (const auto& task : tasks) {
    const int stream_id =
        final_schedule.task_stream.at(task.id);

    if (stream_id < 0 ||
        stream_id >= stream_count) {
      throw std::runtime_error(
          "invalid stream id for task " +
          std::to_string(task.id)
      );
    }

    ++task_count_per_stream.at(
        static_cast<std::size_t>(stream_id)
    );
  }

  std::cout
      << "submission                 : "
      << "single CPU thread, direct CUDA enqueue"
      << '\n';

  std::cout
      << "tasks per stream           : ";

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    if (stream_id > 0) {
      std::cout << ',';
    }

    std::cout
        << task_count_per_stream.at(
               static_cast<std::size_t>(stream_id)
           );
  }

  std::cout << '\n';

  /*
   * gpu_submit_wait_ms:
   *   選択した複数stream構成で実際に実行し、
   *   CUDA commandの投入開始からGPU上の全処理が完了するまでの
   *   実時間を表す。
   *
   * gpu_kernel_ms:
   *   最初のtask kernel開始から最後のtask kernel終了までの
   *   実測GPU makespan。baseline/proposedで同じ定義を使う。
   */
  const auto submit_start = Clock::now();

  /*
   * CPUはGPU上の依存完了を待たず、全commandを順番に投入する。
   *
   * predecessorのeventはトポロジカル順により必ず先にrecord済み。
   * 実際の依存待ちはcudaStreamWaitEventがGPU側で行う。
   */
  for (const TaskSpec* task_pointer : enqueue_order) {
    if (task_pointer == nullptr) {
      throw std::runtime_error(
          "null task in CUDA enqueue order"
      );
    }

    const TaskSpec& task = *task_pointer;

    const int stream_id =
        final_schedule.task_stream.at(task.id);

    cudaStream_t stream =
        get_stream_by_id(
            rr,
            stream_id
        );

    for (const int pred_id : task.preds) {
      const auto event_it =
          done_events.find(pred_id);

      if (event_it == done_events.end()) {
        throw std::runtime_error(
            "unknown predecessor event: " +
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

    float* stream_dmem =
        dmem +
        static_cast<std::size_t>(stream_id) *
        static_cast<std::size_t>(N);

    launch_task_kernel(
        task,
        stream_dmem,
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
  }

  /*
   * 各streamの最後へ完了eventを記録する。
   * このeventが完了したstreamから順にGCを解放する。
   */
  std::vector<cudaEvent_t> stream_completion_events(
      static_cast<std::size_t>(stream_count),
      nullptr
  );

  for (int stream_id = 0;
       stream_id < stream_count;
       ++stream_id) {
    CUDA_CHECK(
        cudaEventCreateWithFlags(
            &stream_completion_events.at(
                static_cast<std::size_t>(stream_id)
            ),
            cudaEventDisableTiming
        )
    );

    CUDA_CHECK(
        cudaEventRecord(
            stream_completion_events.at(
                static_cast<std::size_t>(stream_id)
            ),
            get_stream_by_id(
                rr,
                stream_id
            )
        )
    );
  }

  /*
   * streamを解放する前に、task eventの時刻を回収する。
   */
  double first_kernel_start_ms =
      std::numeric_limits<double>::max();

  double last_kernel_stop_ms = 0.0;
  double measured_stream_overlap_ms = 0.0;

  std::vector<
      std::vector<std::pair<double, double>>
  > intervals_per_stream(
      static_cast<std::size_t>(stream_count)
  );

  std::vector<bool> stream_released(
      static_cast<std::size_t>(stream_count),
      false
  );

  int released_stream_count = 0;
  auto submit_end = submit_start;

  /*
   * cudaDeviceSynchronizeで全streamを一括待機しない。
   *
   * 各stream末尾のeventをcudaEventQueryで確認し、
   * 完了したstreamの計測値を回収して直ちに解放する。
   */
  while (released_stream_count < stream_count) {
    bool released_any_stream = false;

    for (int stream_id = 0;
         stream_id < stream_count;
         ++stream_id) {
      const std::size_t index =
          static_cast<std::size_t>(stream_id);

      if (stream_released.at(index)) {
        continue;
      }

      const cudaError_t query_result =
          cudaEventQuery(
              stream_completion_events.at(index)
          );

      if (query_result == cudaErrorNotReady) {
        continue;
      }

      CUDA_CHECK(query_result);

      /*
       * このstreamは全taskを完了している。
       * GCを破棄する前に、このstream上のkernel時刻を取得する。
       */
      for (const auto& task : tasks) {
        if (final_schedule.task_stream.at(task.id) !=
            stream_id) {
          continue;
        }

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

        intervals_per_stream.at(index).emplace_back(
            static_cast<double>(start_from_origin_ms),
            static_cast<double>(stop_from_origin_ms)
        );
      }

      stream_released.at(index) = true;
      ++released_stream_count;
      released_any_stream = true;

      /*
       * 最後のstreamの完了を観測した時点を、
       * GPU処理全体の終了時刻とする。
       * GC破棄時間はgpu_submit_wait_msへ含めない。
       */
      if (released_stream_count == stream_count) {
        submit_end = Clock::now();
      }

      std::cout
          << "[stream completed] stream["
          << stream_id
          << "]\n";

      release_gc_stream(
          rr,
          stream_id
      );
    }

    if (!released_any_stream) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(100)
      );
    }
  }

  bench.gpu_submit_wait_ms =
      elapsed_ms(
          submit_start,
          submit_end
      );

  /*
   * baseline/proposedの両方で同じ定義を使う。
   *
   * gpu_kernel_ms:
   *   最初のtask kernel開始から、
   *   最後のtask kernel終了までの実測GPU makespan。
   */
  if (tasks.empty()) {
    bench.gpu_kernel_ms = 0.0;
  }
  else {
    bench.gpu_kernel_ms =
        last_kernel_stop_ms -
        first_kernel_start_ms;
  }

  if (stream_count >= 2) {
    std::vector<
        std::vector<std::pair<double, double>>
    > merged_intervals_per_stream(
        static_cast<std::size_t>(stream_count)
    );

    for (int stream_id = 0;
         stream_id < stream_count;
         ++stream_id) {
      merged_intervals_per_stream.at(
          static_cast<std::size_t>(stream_id)
      ) =
          merge_intervals(
              intervals_per_stream.at(
                  static_cast<std::size_t>(stream_id)
              )
          );
    }

    /*
     * 全stream組合せのうち、最も長く重なった時間を表示する。
     */
    for (int lhs_stream_id = 0;
         lhs_stream_id < stream_count;
         ++lhs_stream_id) {
      for (int rhs_stream_id = lhs_stream_id + 1;
           rhs_stream_id < stream_count;
           ++rhs_stream_id) {
        measured_stream_overlap_ms =
            std::max(
                measured_stream_overlap_ms,
                calculate_interval_overlap_ms(
                    merged_intervals_per_stream.at(
                        static_cast<std::size_t>(
                            lhs_stream_id
                        )
                    ),
                    merged_intervals_per_stream.at(
                        static_cast<std::size_t>(
                            rhs_stream_id
                        )
                    )
                )
            );
      }
    }

    std::cout
        << "max_pairwise_overlap_ms    : "
        << measured_stream_overlap_ms
        << " ms\n";

    std::cout
        << "parallel_overlap_detected  : "
        << (
             measured_stream_overlap_ms > 0.0
                 ? "yes"
                 : "no"
           )
        << '\n';
  }

  /*
   * CUDA Eventを破棄
   */
  for (auto& event : stream_completion_events) {
    if (event != nullptr) {
      CUDA_CHECK(
          cudaEventDestroy(event)
      );

      event = nullptr;
    }
  }

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
  if (argc != 3) {
    std::cerr
        << "Usage: "
        << argv[0]
        << " input.stg baseline|proposed|compare\n";

    return 1;
  }

  const std::string mode = argv[2];

  if (mode != "baseline" &&
      mode != "proposed" &&
      mode != "compare") {
    std::cerr
        << "unknown mode: "
        << mode
        << '\n';

    return 1;
  }

  try {
    const StgGraph graph =
        load_stg_without_comm(argv[1]);

    const auto tasks =
        make_task_specs_from_stg(graph);

    print_stg_summary(tasks);

    if (mode == "compare") {
      std::cout
          << "\n===== Baseline run =====\n";

      const BenchResult baseline_result =
          run_cuda_schedule(
              tasks,
              "baseline"
          );

      std::cout << "mode: baseline\n";
      print_result(baseline_result);

      CUDA_CHECK(
          cudaDeviceSynchronize()
      );

      std::cout
          << "\n===== Proposed run =====\n";

      const BenchResult proposed_result =
          run_cuda_schedule(
              tasks,
              "proposed"
          );

      std::cout << "mode: proposed\n";
      print_result(proposed_result);

      const double speedup =
          proposed_result.gpu_kernel_ms > 0.0
              ? baseline_result.gpu_kernel_ms /
                    proposed_result.gpu_kernel_ms
              : 0.0;

      const double reduction_percent =
          baseline_result.gpu_kernel_ms > 0.0
              ? (
                    baseline_result.gpu_kernel_ms -
                    proposed_result.gpu_kernel_ms
                ) /
                    baseline_result.gpu_kernel_ms *
                    100.0
              : 0.0;

      std::cout
          << "\n========== Fair Comparison ==========\n";

      std::cout
          << "baseline_gpu_kernel_ms:     "
          << baseline_result.gpu_kernel_ms
          << " ms\n";

      std::cout
          << "proposed_gpu_kernel_ms:     "
          << proposed_result.gpu_kernel_ms
          << " ms\n";

      std::cout
          << "speedup:                    "
          << speedup
          << " x\n";

      std::cout
          << "time_reduction:             "
          << reduction_percent
          << " %\n";

      std::cout
          << "=====================================\n";
    }
    else {
      const BenchResult result =
          run_cuda_schedule(
              tasks,
              mode
          );

      std::cout
          << "mode: "
          << mode
          << '\n';

      print_result(result);
    }
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
