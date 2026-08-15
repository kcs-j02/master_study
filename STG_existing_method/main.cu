#include <cuda_runtime.h>
#include <taskflow/taskflow.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "bench_timer.hpp"
#include "../stg_common.hpp"
#include "common_types.hpp"
#include "task_DFG_construction.hpp"
#include "task_levelization.hpp"
#include "task_assignment.hpp"
#include "resource_allocation.hpp"

constexpr int kTaskParallelSmLimit = 64;
constexpr int kTaskElementCount =
    kTaskParallelSmLimit * 256;

StgGraph load_stg_without_comm(const std::string& path) {
  return stg::load_stg_without_comm(path);
}

std::vector<TaskSpec> make_task_specs_from_stg(
    const StgGraph& g
) {
  return stg::make_task_specs_from_stg_common<TaskSpec>(
      g,
      [](TaskSpec& s,
         const stg::StgTask& t,
         int threshold) {

        s.parallel_sm_limit =
            kTaskParallelSmLimit;

        if (t.proc_time > threshold) {
          s.kind = KernelKind::HEAVY;

          s.work_units =
              stg::safe_work_units(
                  t.proc_time,
                  200,
                  1000
              );
        }
        else {
          s.kind = KernelKind::LIGHT;

          s.work_units =
              stg::safe_work_units(
                  t.proc_time,
                  80,
                  200
              );
        }
      }
  );
}

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

BenchResult run_taskflow_cuda(
    const std::vector<TaskSpec>& tasks
) {
  BenchResult bench;

  const auto total_start =
      Clock::now();

  TaskDFG dfg;

  {
    ScopedTimer timer(
        bench.task_DFG_construction_ms
    );

    dfg =
        task_DFG_construction(tasks);
  }

  TaskLevels levels;

  {
    ScopedTimer timer(
        bench.task_levelization_ms
    );

    levels =
        task_levelization(dfg);
  }

  constexpr int max_stream_count = 5;
  constexpr int scheduling_reference_sm = 114;

  const int stream_count =
      decide_stream_count(
          levels,
          max_stream_count
      );

  const std::vector<int>
      scheduling_stream_sm_counts(
          static_cast<std::size_t>(
              stream_count
          ),
          scheduling_reference_sm
      );

  StreamScheduleResult final_schedule;

  {
    ScopedTimer timer(
        bench.task_assignment_ms
    );

    final_schedule =
        simulate_stream_assignment(
            tasks,
            scheduling_stream_sm_counts,
            scheduling_reference_sm
        );
  }

  apply_stream_schedule_to_dfg(
      dfg,
      final_schedule
  );

  constexpr int N =
      kTaskElementCount;

  float* dmem = nullptr;

  CUDA_CHECK(
      cudaMalloc(
          &dmem,
          static_cast<std::size_t>(N) *
              static_cast<std::size_t>(
                  stream_count
              ) *
              sizeof(float)
      )
  );

  CUDA_CHECK(
      cudaMemset(
          dmem,
          0,
          static_cast<std::size_t>(N) *
              static_cast<std::size_t>(
                  stream_count
              ) *
              sizeof(float)
      )
  );

  RuntimeResources rr;

  {
    ScopedTimer timer(
        bench.resource_allocation_ms
    );

    rr =
        create_normal_stream_resources(
            stream_count,
            scheduling_reference_sm
        );
  }

  tf::Executor executor(8);
  tf::Taskflow taskflow;

  std::unordered_map<int, tf::Task>
      task_nodes;

  std::unordered_map<int, cudaEvent_t>
      done_events;

  std::unordered_map<int, cudaEvent_t>
      kernel_start_events;

  std::unordered_map<int, cudaEvent_t>
      kernel_stop_events;

  task_nodes.reserve(
      tasks.size()
  );

  done_events.reserve(
      tasks.size()
  );

  kernel_start_events.reserve(
      tasks.size()
  );

  kernel_stop_events.reserve(
      tasks.size()
  );

  cudaStream_t timing_stream =
      nullptr;

  cudaEvent_t origin_event =
      nullptr;

  CUDA_CHECK(
      cudaStreamCreateWithFlags(
          &timing_stream,
          cudaStreamNonBlocking
      )
  );

  CUDA_CHECK(
      cudaEventCreate(
          &origin_event
      )
  );

  CUDA_CHECK(
      cudaEventRecord(
          origin_event,
          timing_stream
      )
  );

  CUDA_CHECK(
      cudaEventSynchronize(
          origin_event
      )
  );

  for (const auto& task : tasks) {

    cudaEvent_t done_event =
        nullptr;

    cudaEvent_t start_event =
        nullptr;

    cudaEvent_t stop_event =
        nullptr;

    CUDA_CHECK(
        cudaEventCreateWithFlags(
            &done_event,
            cudaEventDisableTiming
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

  for (const auto& task : tasks) {

    tf::Task node =
        taskflow.emplace(
            [&, task]() {

              const auto node_it =
                  dfg.nodes.find(
                      task.id
                  );

              if (node_it ==
                  dfg.nodes.end()) {

                throw std::runtime_error(
                    "DFG node not found: " +
                    std::to_string(
                        task.id
                    )
                );
              }

              const int stream_id =
                  node_it
                      ->second
                      .stream_id;

              if (stream_id < 0 ||
                  stream_id >=
                      stream_count) {

                throw std::runtime_error(
                    "invalid stream id"
                );
              }

              cudaStream_t stream =
                  get_stream_by_id(
                      rr,
                      stream_id
                  );

              for (
                  const int pred_id :
                  task.preds
              ) {

                CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        done_events.at(
                            pred_id
                        ),
                        0
                    )
                );
              }

              CUDA_CHECK(
                  cudaEventRecord(
                      kernel_start_events.at(
                          task.id
                      ),
                      stream
                  )
              );

              float* stream_dmem =
                  dmem +
                  static_cast<std::size_t>(
                      stream_id
                  ) *
                  static_cast<std::size_t>(
                      N
                  );

              launch_task_kernel(
                  task,
                  stream_dmem,
                  N,
                  stream
              );

              CUDA_CHECK(
                  cudaEventRecord(
                      kernel_stop_events.at(
                          task.id
                      ),
                      stream
                  )
              );

              CUDA_CHECK(
                  cudaEventRecord(
                      done_events.at(
                          task.id
                      ),
                      stream
                  )
              );
            }
        );

    node.name(
        std::to_string(
            task.id
        )
    );

    task_nodes.emplace(
        task.id,
        node
    );
  }

  for (const auto& task : tasks) {

    for (
        const int pred_id :
        task.preds
    ) {

      task_nodes.at(
          pred_id
      ).precede(
          task_nodes.at(
              task.id
          )
      );
    }
  }

  std::vector<
      std::vector<int>
  > stream_task_order(
      static_cast<std::size_t>(
          stream_count
      )
  );

  std::unordered_map<
      int,
      const TaskSpec*
  > task_spec_by_id;

  task_spec_by_id.reserve(
      tasks.size()
  );

  for (const auto& task : tasks) {

    task_spec_by_id.emplace(
        task.id,
        &task
    );

    const int stream_id =
        final_schedule
            .task_stream
            .at(task.id);

    stream_task_order.at(
        static_cast<std::size_t>(
            stream_id
        )
    ).push_back(
        task.id
    );
  }

  for (
      int stream_id = 0;
      stream_id < stream_count;
      ++stream_id
  ) {

    auto& task_ids =
        stream_task_order.at(
            static_cast<std::size_t>(
                stream_id
            )
        );

    std::stable_sort(
        task_ids.begin(),
        task_ids.end(),
        [&](int lhs_id,
            int rhs_id) {

          const double lhs_start =
              final_schedule
                  .task_start_time
                  .at(lhs_id);

          const double rhs_start =
              final_schedule
                  .task_start_time
                  .at(rhs_id);

          if (lhs_start !=
              rhs_start) {

            return lhs_start <
                   rhs_start;
          }

          const double lhs_finish =
              final_schedule
                  .task_finish_time
                  .at(lhs_id);

          const double rhs_finish =
              final_schedule
                  .task_finish_time
                  .at(rhs_id);

          if (lhs_finish !=
              rhs_finish) {

            return lhs_finish <
                   rhs_finish;
          }

          return lhs_id <
                 rhs_id;
        }
    );

    for (
        std::size_t index = 1;
        index < task_ids.size();
        ++index
    ) {

      const int previous_id =
          task_ids.at(
              index - 1
          );

      const int current_id =
          task_ids.at(
              index
          );

      bool already_predecessor =
          false;

      const TaskSpec* current_task =
          task_spec_by_id.at(
              current_id
          );

      for (
          const int pred_id :
          current_task->preds
      ) {

        if (pred_id ==
            previous_id) {

          already_predecessor =
              true;

          break;
        }
      }

      if (!already_predecessor) {

        task_nodes.at(
            previous_id
        ).precede(
            task_nodes.at(
                current_id
            )
        );
      }
    }
  }

  const auto submit_start =
      Clock::now();

  executor
      .run(taskflow)
      .wait();

  CUDA_CHECK(
      cudaDeviceSynchronize()
  );

  const auto submit_end =
      Clock::now();

  bench.gpu_submit_wait_ms =
      elapsed_ms(
          submit_start,
          submit_end
      );

  if (tasks.empty()) {

    bench.gpu_kernel_ms =
        0.0;
  }
  else {

    double first_start =
        std::numeric_limits<
            double
        >::max();

    double last_stop =
        0.0;

    for (
        const auto& task :
        tasks
    ) {

      float start_ms =
          0.0f;

      float stop_ms =
          0.0f;

      CUDA_CHECK(
          cudaEventElapsedTime(
              &start_ms,
              origin_event,
              kernel_start_events.at(
                  task.id
              )
          )
      );

      CUDA_CHECK(
          cudaEventElapsedTime(
              &stop_ms,
              origin_event,
              kernel_stop_events.at(
                  task.id
              )
          )
      );

      first_start =
          std::min(
              first_start,
              static_cast<double>(
                  start_ms
              )
          );

      last_stop =
          std::max(
              last_stop,
              static_cast<double>(
                  stop_ms
              )
          );
    }

    bench.gpu_kernel_ms =
        last_stop -
        first_start;
  }

  for (
      auto& [task_id, event] :
      done_events
  ) {
    CUDA_CHECK(
        cudaEventDestroy(
            event
        )
    );
  }

  for (
      auto& [task_id, event] :
      kernel_start_events
  ) {
    CUDA_CHECK(
        cudaEventDestroy(
            event
        )
    );
  }

  for (
      auto& [task_id, event] :
      kernel_stop_events
  ) {
    CUDA_CHECK(
        cudaEventDestroy(
            event
        )
    );
  }

  CUDA_CHECK(
      cudaEventDestroy(
          origin_event
      )
  );

  CUDA_CHECK(
      cudaStreamDestroy(
          timing_stream
      )
  );

  destroy_runtime_resources(
      rr
  );

  CUDA_CHECK(
      cudaFree(
          dmem
      )
  );

  const auto total_end =
      Clock::now();

  bench.total_ms =
      elapsed_ms(
          total_start,
          total_end
      );

  return bench;
}

void print_stg_summary(
    const std::vector<TaskSpec>& tasks
) {
  stg::print_stg_summary(
      tasks
  );

  std::cout
      << "task_grid     : "
      << kTaskParallelSmLimit
      << " blocks\n";

  std::cout
      << "task_SM_limit : "
      << kTaskParallelSmLimit
      << " SM\n";
}

int main(
    int argc,
    char** argv
) {
  if (argc != 2) {

    std::cerr
        << "Usage: "
        << argv[0]
        << " input.stg\n";

    return 1;
  }

  try {

    const StgGraph graph =
        load_stg_without_comm(
            argv[1]
        );

    const auto tasks =
        make_task_specs_from_stg(
            graph
        );

    print_stg_summary(
        tasks
    );

    const BenchResult result =
        run_taskflow_cuda(
            tasks
        );

    std::cout
        << "mode: existing_method\n";

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
        << '\n';

    return 1;
  }

  return 0;
}