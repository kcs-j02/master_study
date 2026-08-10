// ============================================================
// Baseline
// STGの依存関係を守り、単一CUDA streamで逐次実行
//
// Build:
// nvcc -O2 -std=c++20 main.cu -I/home/kobayashi/taskflow -o main
//
// Run:
// ./main sample.stg
//
// 比較Pythonからも
// ./main sample.stg
// でそのまま実行可能
// ============================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "bench_timer.hpp"


// ============================================================
// CUDA error check
// ============================================================

#define CUDA_CHECK(expr)                                              \
  do {                                                                \
    const cudaError_t err__ = (expr);                                 \
    if (err__ != cudaSuccess) {                                       \
      std::cerr << "CUDA error: "                                     \
                << cudaGetErrorString(err__)                          \
                << " at "                                             \
                << __FILE__                                           \
                << ":"                                                \
                << __LINE__                                           \
                << "\n";                                              \
      std::exit(EXIT_FAILURE);                                        \
    }                                                                 \
  } while (0)


// ============================================================
// STG
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


// ============================================================
// Task
// ============================================================

enum class KernelKind {
  LIGHT,
  HEAVY
};


struct TaskSpec {
  int id = -1;
  int proc_time = 0;

  KernelKind kind = KernelKind::LIGHT;

  int work_units = 1;

  std::vector<int> preds;
};


// ============================================================
// trim
// ============================================================

static std::string trim(
  const std::string& s
) {
  const std::size_t first =
    s.find_first_not_of(" \t\r\n");

  if (first == std::string::npos) {
    return "";
  }

  const std::size_t last =
    s.find_last_not_of(" \t\r\n");

  return s.substr(
    first,
    last - first + 1
  );
}


// ============================================================
// コメント削除
//
// # comment
// // comment
//
// 両方対応
// ============================================================

static std::string remove_comment(
  const std::string& line
) {
  std::size_t comment_pos =
    std::string::npos;

  const std::size_t hash_pos =
    line.find('#');

  const std::size_t slash_pos =
    line.find("//");

  if (hash_pos != std::string::npos) {
    comment_pos = hash_pos;
  }

  if (slash_pos != std::string::npos) {

    if (comment_pos == std::string::npos) {
      comment_pos = slash_pos;
    }
    else {
      comment_pos =
        std::min(
          comment_pos,
          slash_pos
        );
    }
  }

  return trim(
    line.substr(
      0,
      comment_pos
    )
  );
}


// ============================================================
// STG load
// ============================================================

StgGraph load_stg_without_comm(
  const std::string& path
) {
  std::ifstream ifs(path);

  if (!ifs) {
    throw std::runtime_error(
      "failed to open STG file: " + path
    );
  }

  std::vector<std::string> lines;

  std::string line;

  while (std::getline(ifs, line)) {

    line = remove_comment(line);

    if (!line.empty()) {
      lines.push_back(line);
    }
  }


  if (lines.empty()) {
    throw std::runtime_error(
      "empty STG file: " + path
    );
  }


  StgGraph graph;


  // ----------------------------------------------------------
  // タスク数
  // ----------------------------------------------------------

  {
    std::istringstream iss(
      lines.front()
    );

    if (!(iss >> graph.num_tasks) ||
        graph.num_tasks < 0) {

      throw std::runtime_error(
        "failed to parse number of tasks"
      );
    }
  }


  if (
    static_cast<int>(lines.size())
    <
    graph.num_tasks + 1
  ) {
    throw std::runtime_error(
      "STG file has fewer task lines than expected"
    );
  }


  graph.tasks.reserve(
    graph.num_tasks
  );


  // ----------------------------------------------------------
  // 各タスク読み込み
  // ----------------------------------------------------------

  for (
    int i = 0;
    i < graph.num_tasks;
    ++i
  ) {

    std::istringstream iss(
      lines[i + 1]
    );

    StgTask task;

    int pred_count = 0;


    if (!(
      iss
      >> task.id
      >> task.proc_time
      >> pred_count
    )) {
      throw std::runtime_error(
        "failed to parse task header"
      );
    }


    if (task.id < 0) {
      throw std::runtime_error(
        "negative task id"
      );
    }


    if (task.proc_time < 0) {
      throw std::runtime_error(
        "negative processing time at task "
        + std::to_string(task.id)
      );
    }


    if (pred_count < 0) {
      throw std::runtime_error(
        "negative predecessor count at task "
        + std::to_string(task.id)
      );
    }


    task.preds.resize(
      pred_count
    );


    for (
      int k = 0;
      k < pred_count;
      ++k
    ) {

      if (!(iss >> task.preds[k])) {

        throw std::runtime_error(
          "failed to parse predecessor list at task "
          + std::to_string(task.id)
        );
      }
    }


    graph.tasks.push_back(
      std::move(task)
    );
  }


  // ----------------------------------------------------------
  // 重複ID確認
  // ----------------------------------------------------------

  std::vector<int> ids;

  ids.reserve(
    graph.tasks.size()
  );


  for (const auto& task : graph.tasks) {
    ids.push_back(task.id);
  }


  std::sort(
    ids.begin(),
    ids.end()
  );


  if (
    std::adjacent_find(
      ids.begin(),
      ids.end()
    )
    != ids.end()
  ) {
    throw std::runtime_error(
      "duplicate task ids found in STG"
    );
  }


  return graph;
}


// ============================================================
// work_units
//
// overflow防止
// ============================================================

static int safe_work_units(
  int proc_time,
  int scale,
  int minimum
) {
  const long long value =
    static_cast<long long>(
      std::max(
        1,
        proc_time
      )
    )
    * scale;


  return static_cast<int>(

    std::min<long long>(

      std::max<long long>(
        minimum,
        value
      ),

      std::numeric_limits<int>::max()
    )
  );
}


// ============================================================
// STG -> TaskSpec
// ============================================================

std::vector<TaskSpec>
make_task_specs_from_stg(
  const StgGraph& graph
) {
  std::vector<int> positive_times;

  positive_times.reserve(
    graph.tasks.size()
  );


  for (const auto& task : graph.tasks) {

    if (task.proc_time > 0) {
      positive_times.push_back(
        task.proc_time
      );
    }
  }


  // ----------------------------------------------------------
  // 中央値をLIGHT / HEAVYの閾値にする
  // ----------------------------------------------------------

  int threshold = 1;


  if (!positive_times.empty()) {

    std::sort(
      positive_times.begin(),
      positive_times.end()
    );

    threshold =
      positive_times[
        positive_times.size() / 2
      ];
  }


  std::vector<TaskSpec> specs;

  specs.reserve(
    graph.tasks.size()
  );


  for (const auto& task : graph.tasks) {

    TaskSpec spec;

    spec.id =
      task.id;

    spec.proc_time =
      task.proc_time;

    spec.preds =
      task.preds;


    if (
      task.proc_time
      >
      threshold
    ) {

      spec.kind =
        KernelKind::HEAVY;

      spec.work_units =
        safe_work_units(
          task.proc_time,
          200,
          1000
        );
    }
    else {

      spec.kind =
        KernelKind::LIGHT;

      spec.work_units =
        safe_work_units(
          task.proc_time,
          80,
          200
        );
    }


    specs.push_back(
      std::move(spec)
    );
  }


  return specs;
}


// ============================================================
// GPU Kernel
// ============================================================

__global__
void light_kernel(
  float* data,
  int n,
  int iters
) {
  const int idx =
    blockIdx.x
    * blockDim.x
    + threadIdx.x;


  if (idx >= n) {
    return;
  }


  float x =
    data[idx];


#pragma unroll 1
  for (
    int i = 0;
    i < iters;
    ++i
  ) {

    x =
      x * 1.000001f
      + 0.00001f;
  }


  data[idx] =
    x;
}


// ============================================================

__global__
void heavy_kernel(
  float* data,
  int n,
  int iters
) {
  const int idx =
    blockIdx.x
    * blockDim.x
    + threadIdx.x;


  if (idx >= n) {
    return;
  }


  float x =
    data[idx];


#pragma unroll 1
  for (
    int i = 0;
    i < iters;
    ++i
  ) {

    x =
      x * 1.000001f
      + 0.00001f;

    x =
      x * 0.999999f
      + 0.00002f;

    x =
      x * 1.0000003f
      - 0.00001f;
  }


  data[idx] =
    x;
}


// ============================================================
// Kernel launch
// ============================================================

void launch_task_kernel(
  const TaskSpec& task,
  float* dmem,
  int n,
  cudaStream_t stream
) {
  constexpr int THREADS = 256;

  const int blocks =
    (
      n
      + THREADS
      - 1
    )
    /
    THREADS;


  if (
    task.kind
    ==
    KernelKind::HEAVY
  ) {

    heavy_kernel
    <<<blocks, THREADS, 0, stream>>>(
      dmem,
      n,
      task.work_units
    );
  }
  else {

    light_kernel
    <<<blocks, THREADS, 0, stream>>>(
      dmem,
      n,
      task.work_units
    );
  }


  CUDA_CHECK(
    cudaGetLastError()
  );
}


// ============================================================
// DAGをトポロジカルソート
//
// 依存関係を守った実行順を作る
//
// 同時に実行可能なタスクは
// task IDの小さい順
// ============================================================

std::vector<std::size_t>
make_topological_order(
  const std::vector<TaskSpec>& tasks
) {
  std::unordered_map<
    int,
    std::size_t
  > index_by_id;


  index_by_id.reserve(
    tasks.size()
  );


  // ----------------------------------------------------------
  // id -> vector index
  // ----------------------------------------------------------

  for (
    std::size_t i = 0;
    i < tasks.size();
    ++i
  ) {

    if (
      !index_by_id.emplace(
        tasks[i].id,
        i
      ).second
    ) {

      throw std::runtime_error(
        "duplicate task id: "
        + std::to_string(
          tasks[i].id
        )
      );
    }
  }


  // ----------------------------------------------------------
  // indegree / successors
  // ----------------------------------------------------------

  std::vector<int> indegree(
    tasks.size(),
    0
  );


  std::vector<
    std::vector<std::size_t>
  > successors(
    tasks.size()
  );


  for (
    std::size_t i = 0;
    i < tasks.size();
    ++i
  ) {

    indegree[i] =
      static_cast<int>(
        tasks[i].preds.size()
      );


    for (
      const int pred_id
      :
      tasks[i].preds
    ) {

      const auto pred_it =
        index_by_id.find(
          pred_id
        );


      if (
        pred_it
        ==
        index_by_id.end()
      ) {

        throw std::runtime_error(
          "unknown predecessor id "
          + std::to_string(pred_id)
          + " referenced by task "
          + std::to_string(
            tasks[i].id
          )
        );
      }


      successors[
        pred_it->second
      ].push_back(i);
    }
  }


  // ----------------------------------------------------------
  // ready queue
  //
  // IDが小さいものを優先
  // ----------------------------------------------------------

  using ReadyItem =
    std::pair<
      int,
      std::size_t
    >;


  std::priority_queue<
    ReadyItem,
    std::vector<ReadyItem>,
    std::greater<ReadyItem>
  > ready;


  for (
    std::size_t i = 0;
    i < tasks.size();
    ++i
  ) {

    if (
      indegree[i]
      ==
      0
    ) {

      ready.emplace(
        tasks[i].id,
        i
      );
    }
  }


  // ----------------------------------------------------------
  // Topological order
  // ----------------------------------------------------------

  std::vector<std::size_t> order;

  order.reserve(
    tasks.size()
  );


  while (
    !ready.empty()
  ) {

    const std::size_t current =
      ready.top().second;

    ready.pop();


    order.push_back(
      current
    );


    for (
      const std::size_t next
      :
      successors[current]
    ) {

      --indegree[next];


      if (
        indegree[next]
        ==
        0
      ) {

        ready.emplace(
          tasks[next].id,
          next
        );
      }
    }
  }


  // ----------------------------------------------------------
  // cycle check
  // ----------------------------------------------------------

  if (
    order.size()
    !=
    tasks.size()
  ) {

    throw std::runtime_error(
      "cycle detected in STG graph"
    );
  }


  return order;
}


// ============================================================
// Baseline execution
//
// 依存関係解析
//       ↓
// トポロジカルソート
//       ↓
// 単一CUDA stream
//       ↓
// 全タスク逐次実行
// ============================================================

BenchResult run_sequential_cuda(
  const std::vector<TaskSpec>& tasks
) {
  BenchResult bench{};


  // ==========================================================
  // total
  // ==========================================================

  const auto total_start =
    Clock::now();


  // ==========================================================
  // Levelization / DAG order
  // ==========================================================

  const auto levelization_start =
    Clock::now();


  const std::vector<std::size_t> order =
    make_topological_order(
      tasks
    );


  const auto levelization_end =
    Clock::now();


  // ==========================================================
  // GPU resource allocation
  // ==========================================================

  constexpr int N =
    1 << 20;


  float* dmem =
    nullptr;


  cudaStream_t stream{};


  cudaEvent_t kernel_start{};
  cudaEvent_t kernel_stop{};


  const auto resource_start =
    Clock::now();


  CUDA_CHECK(
    cudaMalloc(
      &dmem,
      N * sizeof(float)
    )
  );


  CUDA_CHECK(
    cudaStreamCreateWithFlags(
      &stream,
      cudaStreamNonBlocking
    )
  );


  CUDA_CHECK(
    cudaEventCreate(
      &kernel_start
    )
  );


  CUDA_CHECK(
    cudaEventCreate(
      &kernel_stop
    )
  );


  CUDA_CHECK(
    cudaMemsetAsync(
      dmem,
      0,
      N * sizeof(float),
      stream
    )
  );


  CUDA_CHECK(
    cudaStreamSynchronize(
      stream
    )
  );


  const auto resource_end =
    Clock::now();


  // ==========================================================
  // GPU submit + wait
  //
  // この時間を比較Pythonで使用
  // ==========================================================

  const auto submit_start =
    Clock::now();


  // ----------------------------------------------------------
  // GPU実行開始イベント
  // ----------------------------------------------------------

  CUDA_CHECK(
    cudaEventRecord(
      kernel_start,
      stream
    )
  );


  // ----------------------------------------------------------
  // トポロジカル順に単一streamへ投入
  //
  // 同じstreamなので完全逐次
  // ----------------------------------------------------------

  for (
    const std::size_t task_index
    :
    order
  ) {

    const TaskSpec& task =
      tasks[
        task_index
      ];


    launch_task_kernel(
      task,
      dmem,
      N,
      stream
    );
  }


  // ----------------------------------------------------------
  // 最終Kernelの後ろ
  // ----------------------------------------------------------

  CUDA_CHECK(
    cudaEventRecord(
      kernel_stop,
      stream
    )
  );


  // ----------------------------------------------------------
  // GPU完了待ち
  // ----------------------------------------------------------

  CUDA_CHECK(
    cudaEventSynchronize(
      kernel_stop
    )
  );


  const auto submit_end =
    Clock::now();


  // ==========================================================
  // gpu_kernel_ms
  // ==========================================================

  float kernel_ms =
    0.0f;


  CUDA_CHECK(
    cudaEventElapsedTime(
      &kernel_ms,
      kernel_start,
      kernel_stop
    )
  );


  // ==========================================================
  // BenchResult
  // ==========================================================

  bench.gpu_submit_wait_ms =
    elapsed_ms(
      submit_start,
      submit_end
    );


  bench.gpu_kernel_ms =
    static_cast<double>(
      kernel_ms
    );


  // baselineでは特別なDFG構築処理なし
  bench.task_DFG_construction_ms =
    0.0;


  // トポロジカルソート時間
  bench.task_levelization_ms =
    elapsed_ms(
      levelization_start,
      levelization_end
    );


  // stream割当なし
  bench.task_assignment_ms =
    0.0;


  // malloc / stream / event作成
  bench.resource_allocation_ms =
    elapsed_ms(
      resource_start,
      resource_end
    );


  // ==========================================================
  // Cleanup
  // ==========================================================

  CUDA_CHECK(
    cudaEventDestroy(
      kernel_start
    )
  );


  CUDA_CHECK(
    cudaEventDestroy(
      kernel_stop
    )
  );


  CUDA_CHECK(
    cudaStreamDestroy(
      stream
    )
  );


  CUDA_CHECK(
    cudaFree(
      dmem
    )
  );


  // ==========================================================
  // total
  // ==========================================================

  const auto total_end =
    Clock::now();


  bench.total_ms =
    elapsed_ms(
      total_start,
      total_end
    );


  return bench;
}


// ============================================================
// STG summary
// ============================================================

void print_stg_summary(
  const std::vector<TaskSpec>& tasks
) {
  int light_count =
    0;

  int heavy_count =
    0;

  long long total_proc_time =
    0;


  for (
    const auto& task
    :
    tasks
  ) {

    total_proc_time +=
      task.proc_time;


    if (
      task.kind
      ==
      KernelKind::HEAVY
    ) {

      ++heavy_count;
    }
    else {

      ++light_count;
    }
  }


  std::cout
    << "===== STG summary =====\n"

    << "num_tasks     : "
    << tasks.size()
    << "\n"

    << "light_tasks   : "
    << light_count
    << "\n"

    << "heavy_tasks   : "
    << heavy_count
    << "\n"

    << "sum_proc_time : "
    << total_proc_time
    << "\n";
}


// ============================================================
// main
// ============================================================

int main(
  int argc,
  char** argv
) {
  // ----------------------------------------------------------
  // 比較Pythonとの互換性のため
  //
  // ./main sample.stg
  //
  // で実行できる
  // ----------------------------------------------------------

  if (argc < 2) {

    std::cerr
      << "Usage: "
      << argv[0]
      << " input.stg\n";

    return EXIT_FAILURE;
  }


  try {

    // --------------------------------------------------------
    // STG load
    // --------------------------------------------------------

    const StgGraph graph =
      load_stg_without_comm(
        argv[1]
      );


    // --------------------------------------------------------
    // Task生成
    // --------------------------------------------------------

    const std::vector<TaskSpec> tasks =
      make_task_specs_from_stg(
        graph
      );


    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------

    print_stg_summary(
      tasks
    );


    // --------------------------------------------------------
    // Baseline
    // --------------------------------------------------------

    const BenchResult result =
      run_sequential_cuda(
        tasks
      );


    // --------------------------------------------------------
    // 出力
    //
    // 比較Pythonのparse_result()と互換
    // --------------------------------------------------------

    std::cout
      << "mode: baseline\n";


    print_result(
      result
    );
  }

  catch (
    const std::exception& e
  ) {

    std::cerr
      << "exception: "
      << e.what()
      << "\n";

    return EXIT_FAILURE;
  }


  return EXIT_SUCCESS;
}