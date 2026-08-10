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
#include "../stg_common.hpp"

using StgTask = stg::StgTask;
using StgGraph = stg::StgGraph;
using KernelKind = stg::KernelKind;
using TaskSpec = stg::TaskSpec;


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
// STG load / task spec generation
// ============================================================

StgGraph load_stg_without_comm(
  const std::string& path
) {
  return stg::load_stg_without_comm(path);
}


static int safe_work_units(
  int proc_time,
  int scale,
  int minimum
) {
  return stg::safe_work_units(
    proc_time,
    scale,
    minimum
  );
}


std::vector<TaskSpec> make_task_specs_from_stg(
  const StgGraph& graph
) {
  return stg::make_task_specs_from_stg_common<TaskSpec>(
    graph,
    [](
      TaskSpec& spec,
      const stg::StgTask& task,
      int threshold
    ) {
      if (task.proc_time > threshold) {

        spec.kind =
          KernelKind::HEAVY;

        spec.work_units =
          stg::safe_work_units(
            task.proc_time,
            200,
            1000
          );
      }
      else {

        spec.kind =
          KernelKind::LIGHT;

        spec.work_units =
          stg::safe_work_units(
            task.proc_time,
            80,
            200
          );
      }
    }
  );
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
    blockIdx.x *
    blockDim.x +
    threadIdx.x;

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
    blockIdx.x *
    blockDim.x +
    threadIdx.x;

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
  stg::launch_task_kernel(
    task,
    dmem,
    n,
    stream
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
  return stg::make_topological_order(
    tasks
  );
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
  // ----------------------------------------------------------
  // 既存のサマリ
  // ----------------------------------------------------------

  stg::print_stg_summary(
    tasks
  );


  // ----------------------------------------------------------
  // 全タスクの proc_time の総和
  // ----------------------------------------------------------

  long long total_task_proc_time =
    0;


  for (
    const auto& task
    :
    tasks
  ) {

    total_task_proc_time +=
      static_cast<long long>(
        task.proc_time
      );
  }


  std::cout
    << "total task proc_time : "
    << total_task_proc_time
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