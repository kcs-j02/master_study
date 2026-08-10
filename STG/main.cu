// ============================================================
// Baseline
//
// 提案手法と同じタスク・同じKernel条件で、
// 単一CUDA streamによる逐次実行を測定する。
//
// 評価指標:
//   gpu_kernel_ms
//
// Build:
//   nvcc -O2 -std=c++20 main.cu \
//     -I/home/kobayashi/taskflow \
//     -o main
//
// Run:
//   ./main ../common_sample.stg
// ============================================================

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "bench_timer.hpp"
#include "../stg_common.hpp"


using StgTask   = stg::StgTask;
using StgGraph  = stg::StgGraph;
using KernelKind = stg::KernelKind;
using TaskSpec  = stg::TaskSpec;


// ============================================================
// CUDA error check
// ============================================================

#define CUDA_CHECK(expr)                                      \
  do {                                                        \
    const cudaError_t err__ = (expr);                         \
    if (err__ != cudaSuccess) {                               \
      std::cerr                                               \
        << "CUDA error: "                                     \
        << cudaGetErrorString(err__)                          \
        << " at "                                             \
        << __FILE__                                           \
        << ":"                                                \
        << __LINE__                                           \
        << "\n";                                              \
      std::exit(EXIT_FAILURE);                                \
    }                                                         \
  } while (0)


// ============================================================
// 提案手法と完全に同じKernelサイズ
//
// 1 task = 最大64 blocks
// 1 block = 256 threads
// ============================================================

constexpr int kTaskParallelSmLimit = 64;

constexpr int kTaskElementCount =
    kTaskParallelSmLimit * 256;


// ============================================================
// STG load
// ============================================================

StgGraph load_stg_without_comm(
    const std::string& path
) {
  return stg::load_stg_without_comm(
      path
  );
}


// ============================================================
// TaskSpec生成
//
// 提案手法と同じ条件
// ============================================================

std::vector<TaskSpec>
make_task_specs_from_stg(
    const StgGraph& graph
) {
  return
    stg::make_task_specs_from_stg_common<TaskSpec>(
      graph,

      [](
          TaskSpec& spec,
          const stg::StgTask& task,
          int threshold
      ) {

        // 提案手法と同じ
        spec.parallel_sm_limit =
            kTaskParallelSmLimit;


        // ----------------------------------------------------
        // HEAVY
        // ----------------------------------------------------

        if (
          task.proc_time >
          threshold
        ) {

          spec.kind =
              KernelKind::HEAVY;

          spec.work_units =
              stg::safe_work_units(
                  task.proc_time,
                  200,
                  1000
              );
        }

        // ----------------------------------------------------
        // LIGHT
        // ----------------------------------------------------

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
// Kernel launch
//
// stg_common.hpp の共通Kernelを使用するため、
// proposed と baseline でKernel自体も同じ。
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
// トポロジカル順序
// ============================================================

std::vector<std::size_t>
make_topological_order(
    const std::vector<TaskSpec>& tasks
) {
  return
    stg::make_topological_order(
        tasks
    );
}


// ============================================================
// Baseline逐次実行
//
// 1 stream
// ↓
// 全タスクを依存関係順に投入
// ↓
// CUDA Eventで
// 最初のKernel開始～最後のKernel終了
// を測定
//
// proposed の gpu_kernel_ms と同じ指標
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
  // DAG順序生成
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
  // proposed と同じ task size
  // ==========================================================

  constexpr int N =
      kTaskElementCount;


  float* dmem =
      nullptr;


  cudaStream_t stream =
      nullptr;


  cudaEvent_t start_event =
      nullptr;


  cudaEvent_t stop_event =
      nullptr;


  // ==========================================================
  // resource allocation
  // ==========================================================

  const auto resource_start =
      Clock::now();


  CUDA_CHECK(
      cudaMalloc(
          &dmem,
          static_cast<std::size_t>(N)
              * sizeof(float)
      )
  );


  CUDA_CHECK(
      cudaMemset(
          dmem,
          0,
          static_cast<std::size_t>(N)
              * sizeof(float)
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
          &start_event
      )
  );


  CUDA_CHECK(
      cudaEventCreate(
          &stop_event
      )
  );


  const auto resource_end =
      Clock::now();


  // ==========================================================
  // GPU submit + wait
  //
  // これは参考値として残す。
  // 高速化率の比較には gpu_kernel_ms を使用する。
  // ==========================================================

  const auto submit_start =
      Clock::now();


  // ----------------------------------------------------------
  // 最初のKernelの直前
  // ----------------------------------------------------------

  CUDA_CHECK(
      cudaEventRecord(
          start_event,
          stream
      )
  );


  // ----------------------------------------------------------
  // 1 streamへ全タスク投入
  //
  // 同一streamなので完全逐次
  // ----------------------------------------------------------

  for (
      const std::size_t task_index
      :
      order
  ) {

    const TaskSpec& task =
        tasks.at(
            task_index
        );


    launch_task_kernel(
        task,
        dmem,
        N,
        stream
    );
  }


  // ----------------------------------------------------------
  // 最後のKernelの直後
  // ----------------------------------------------------------

  CUDA_CHECK(
      cudaEventRecord(
          stop_event,
          stream
      )
  );


  // ----------------------------------------------------------
  // 全Kernel終了待ち
  // ----------------------------------------------------------

  CUDA_CHECK(
      cudaEventSynchronize(
          stop_event
      )
  );


  const auto submit_end =
      Clock::now();


  // ==========================================================
  // proposedと同じgpu_kernel_ms
  // ==========================================================

  float gpu_kernel_ms =
      0.0f;


  CUDA_CHECK(
      cudaEventElapsedTime(
          &gpu_kernel_ms,
          start_event,
          stop_event
      )
  );


  // ==========================================================
  // Result
  // ==========================================================

  bench.gpu_kernel_ms =
      static_cast<double>(
          gpu_kernel_ms
      );


  bench.gpu_submit_wait_ms =
      elapsed_ms(
          submit_start,
          submit_end
      );


  bench.task_DFG_construction_ms =
      0.0;


  bench.task_levelization_ms =
      elapsed_ms(
          levelization_start,
          levelization_end
      );


  bench.task_assignment_ms =
      0.0;


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
          start_event
      )
  );


  CUDA_CHECK(
      cudaEventDestroy(
          stop_event
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
// Summary
// ============================================================

void print_stg_summary(
    const std::vector<TaskSpec>& tasks
) {
  stg::print_stg_summary(
      tasks
  );


  std::cout
      << "task_grid      : "
      << kTaskParallelSmLimit
      << " blocks\n";


  std::cout
      << "task_SM_limit  : "
      << kTaskParallelSmLimit
      << " SM\n";
}


// ============================================================
// main
// ============================================================

int main(
    int argc,
    char** argv
) {
  if (argc < 2) {

    std::cerr
        << "Usage: "
        << argv[0]
        << " input.stg\n";

    return EXIT_FAILURE;
  }


  try {

    // ========================================================
    // STG
    // ========================================================

    const StgGraph graph =
        load_stg_without_comm(
            argv[1]
        );


    // ========================================================
    // TaskSpec
    // ========================================================

    const std::vector<TaskSpec> tasks =
        make_task_specs_from_stg(
            graph
        );


    // ========================================================
    // Summary
    // ========================================================

    print_stg_summary(
        tasks
    );


    // ========================================================
    // Sequential baseline
    // ========================================================

    const BenchResult result =
        run_sequential_cuda(
            tasks
        );


    // ========================================================
    // 比較用の値
    //
    // proposed側の
    //
    // proposed gpu_kernel_ms
    //
    // と直接比較する値
    // ========================================================

    std::cout
        << "===== Baseline GPU measurement =====\n";


    std::cout
        << "baseline gpu_kernel_ms: "
        << result.gpu_kernel_ms
        << " ms\n";


    std::cout
        << "====================================\n";


    std::cout
        << "mode: baseline\n";


    print_result(
        result
    );
  }

  catch (
      const std::exception& error
  ) {

    std::cerr
        << "exception: "
        << error.what()
        << "\n";

    return EXIT_FAILURE;
  }


  return EXIT_SUCCESS;
}