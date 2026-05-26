#pragma once
#include <string>
#include <vector>

struct NodeInfo {
  std::string name;
  int indeg = 0;
  std::vector<std::string> preds;
  int stream_id = -1;
};

struct BenchResult {
  double make_levels_ms = 0.0;
  double assign_stream_ms = 0.0;
  double green_ctx_ms = 0.0;
  double gpu_submit_wait_ms = 0.0;
  float  gpu_kernel_ms = 0.0f;
};