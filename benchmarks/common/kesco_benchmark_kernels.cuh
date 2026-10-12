#pragma once

// CUDA kernels adapted from the official grCUDA benchmark suite, commit
// ee0dd4cb1bade06b25ddd7dc56b4441e01172f0e, files b1/b5/b6/b7/b8/b10.
// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano.
// Distributed under the upstream BSD 3-Clause license. See
// benchmarks/common/KESCO_LICENSE.txt.

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace stg::kesco {

inline constexpr int kBlocks = 64;
inline constexpr int kThreads = 32;
inline constexpr int kBlackScholesThreads = 1024;
inline constexpr int kImageThreads2d = 16;
inline constexpr int kDlThreads2d = 8;

// Smallest official V100 problem sizes. They are fixed for every method.
inline constexpr int kVectorElements = 160000000;
inline constexpr int kBlackScholesElements = 10000000;
inline constexpr int kMlRows = 1600000;
inline constexpr int kHitsRows = 25000000;
inline constexpr int kImageWidth = 6400;
inline constexpr int kDlWidth = 12000;

template <typename T>
__global__ void initialize_array(T* values, std::size_t count, T scale) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count; i += blockDim.x * gridDim.x) {
    values[i] = static_cast<T>((i % 97) + 1) * scale;
  }
}

__global__ void initialize_csr(int* row, int* col, int* val, int n, int degree) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i <= n; i += blockDim.x * gridDim.x) {
    row[i] = i * degree;
  }
  const int nnz = n * degree;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < nnz; i += blockDim.x * gridDim.x) {
    const int source = i / degree;
    const int edge = i % degree;
    col[i] = (source * 17 + edge * 13 + 1) % n;
    val[i] = 1;
  }
}

__device__ inline float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void vector_square(const float* x, float* y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) {
    y[i] = x[i] * x[i];
  }
}

__global__ void vector_reduce(const float* x, const float* y, float* z, int n) {
  float value = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) {
    value += x[i] - y[i];
  }
  value = warp_sum(value);
  if ((threadIdx.x & 31) == 0) atomicAdd(z, value);
}

__device__ inline double cumulative_normal(double d) {
  constexpr double a1 = 0.31938153;
  constexpr double a2 = -0.356563782;
  constexpr double a3 = 1.781477937;
  constexpr double a4 = -1.821255978;
  constexpr double a5 = 1.330274429;
  constexpr double inv_sqrt_2pi = 0.3989422804014327;
  const double k = 1.0 / (1.0 + 0.2316419 * fabs(d));
  double cnd = inv_sqrt_2pi * exp(-0.5 * d * d) *
      (k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5)))));
  return d > 0.0 ? 1.0 - cnd : cnd;
}

__global__ void black_scholes(const double* x, double* y, int n) {
  constexpr double r = 0.08;
  constexpr double v = 0.3;
  constexpr double t = 1.0;
  constexpr double strike = 60.0;
  const double sqrt_t = 1.0 / rsqrt(t);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) {
    const double d1 = (log(x[i] / strike) + (r + 0.5 * v * v) * t) /
        (v * sqrt_t);
    const double d2 = d1 - v * sqrt_t;
    y[i] = x[i] * cumulative_normal(d1) -
        strike * exp(-r * t) * cumulative_normal(d2);
  }
}

__global__ void ml_rr_scale(const int* x, float* y, int rows, int features) {
  for (int feature = blockIdx.x * blockDim.x + threadIdx.x;
       feature < features; feature += blockDim.x * gridDim.x) {
    float mean = 0.0f;
    float sum_sq = 0.0f;
    for (int row = 0; row < rows; ++row) {
      const float value = static_cast<float>(x[feature * rows + row]);
      mean += value;
      sum_sq += value * value;
    }
    mean /= rows;
    const float deviation = sqrtf(fmaxf(sum_sq / rows - mean * mean, 1.0e-12f));
    for (int row = 0; row < rows; ++row) {
      y[feature * rows + row] =
          (static_cast<float>(x[feature * rows + row]) - mean) / deviation;
    }
  }
}

template <typename InputT>
__global__ void ml_matmul(const InputT* x, const float* weights, float* output,
                          int rows, int features, int classes) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows; row += blockDim.x * gridDim.x) {
    for (int klass = 0; klass < classes; ++klass) {
      float value = 0.0f;
      for (int feature = 0; feature < features; ++feature) {
        value += static_cast<float>(x[row * features + feature]) *
            weights[klass * features + feature];
      }
      output[row * classes + klass] += value;
    }
  }
}

__global__ void ml_row_max(const float* x, float* output, int rows, int cols) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows; row += blockDim.x * gridDim.x) {
    float value = x[row * cols];
    for (int col = 1; col < cols; ++col) value = fmaxf(value, x[row * cols + col]);
    output[row] = value;
  }
}

__global__ void ml_logsumexp(const float* x, const float* maximum, float* output,
                             int rows, int cols) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows; row += blockDim.x * gridDim.x) {
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
      sum += expf(x[row * cols + col] - maximum[row]);
    }
    output[row] = logf(sum) + maximum[row];
  }
}

__global__ void ml_add_intercept(float* x, const float* intercept,
                                 int rows, int cols) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows; row += blockDim.x * gridDim.x) {
    for (int col = 0; col < cols; ++col) x[row * cols + col] += intercept[col];
  }
}

__global__ void ml_normalize(float* x, const float* normalizer, int rows, int cols) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows; row += blockDim.x * gridDim.x) {
    for (int col = 0; col < cols; ++col) {
      x[row * cols + col] = expf(x[row * cols + col] - normalizer[row]);
    }
  }
}

__global__ void ml_softmax(float* x, int rows, int cols) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows; row += blockDim.x * gridDim.x) {
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) sum += expf(x[row * cols + col]);
    for (int col = 0; col < cols; ++col) x[row * cols + col] = expf(x[row * cols + col]) / sum;
  }
}

__global__ void ml_argmax(const float* x, const float* y, int* output,
                          int rows, int cols) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows; row += blockDim.x * gridDim.x) {
    int best_index = 0;
    float best = x[row * cols] + y[row * cols];
    for (int col = 1; col < cols; ++col) {
      const float value = x[row * cols + col] + y[row * cols + col];
      if (value > best) { best = value; best_index = col; }
    }
    output[row] = best_index;
  }
}

__global__ void hits_spmv(const int* row, const int* col, const int* val,
                          const float* vector, float* output, int rows) {
  for (int r = blockIdx.x * blockDim.x + threadIdx.x;
       r < rows; r += blockDim.x * gridDim.x) {
    float sum = 0.0f;
    for (int i = row[r]; i < row[r + 1]; ++i) sum += val[i] * vector[col[i]];
    output[r] = sum;
  }
}

__global__ void hits_sum(const float* x, float* output, int n) {
  float value = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) value += x[i];
  value = warp_sum(value);
  if ((threadIdx.x & 31) == 0) atomicAdd(output, value);
}

__global__ void hits_divide(const float* x, float* y, const float* norm, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) y[i] = x[i] / norm[0];
}

__global__ void hits_reset(float* a, float* h) {
  if (blockIdx.x == 0 && threadIdx.x == 0) { *a = 0.0f; *h = 0.0f; }
}

__global__ void image_blur(const float* image, float* output, int width,
                           const float* kernel, int diameter) {
  extern __shared__ float local_kernel[];
  const int local = threadIdx.y * blockDim.x + threadIdx.x;
  const int local_count = blockDim.x * blockDim.y;
  for (int i = local; i < diameter * diameter; i += local_count) local_kernel[i] = kernel[i];
  __syncthreads();
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < width; row += blockDim.x * gridDim.x) {
    for (int col = blockIdx.y * blockDim.y + threadIdx.y;
         col < width; col += blockDim.y * gridDim.y) {
      float sum = 0.0f;
      const int radius = diameter / 2;
      for (int x = -radius; x <= radius; ++x) {
        for (int y = -radius; y <= radius; ++y) {
          const int nx = row + x;
          const int ny = col + y;
          if (nx >= 0 && ny >= 0 && nx < width && ny < width) {
            sum += local_kernel[(x + radius) * diameter + y + radius] * image[nx * width + ny];
          }
        }
      }
      output[row * width + col] = sum;
    }
  }
}

__global__ void image_sobel(const float* image, float* output, int width) {
  constexpr int sx[9] = {-1,-2,-1,0,0,0,1,2,1};
  constexpr int sy[9] = {-1,0,1,-2,0,2,-1,0,1};
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < width; row += blockDim.x * gridDim.x) {
    for (int col = blockIdx.y * blockDim.y + threadIdx.y;
         col < width; col += blockDim.y * gridDim.y) {
      float gx = 0.0f, gy = 0.0f;
      for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) {
        const int nx = row + x, ny = col + y;
        if (nx >= 0 && ny >= 0 && nx < width && ny < width) {
          const float pixel = image[nx * width + ny];
          const int k = (x + 1) * 3 + y + 1;
          gx += sx[k] * pixel; gy += sy[k] * pixel;
        }
      }
      output[row * width + col] = sqrtf(gx * gx + gy * gy);
    }
  }
}

__device__ inline float atomic_min_float(float* address, float value) {
  int* bits = reinterpret_cast<int*>(address);
  int old = *bits;
  while (value < __int_as_float(old)) {
    const int assumed = old;
    old = atomicCAS(bits, assumed, __float_as_int(value));
    if (old == assumed) break;
  }
  return __int_as_float(old);
}

__device__ inline float atomic_max_float(float* address, float value) {
  int* bits = reinterpret_cast<int*>(address);
  int old = *bits;
  while (value > __int_as_float(old)) {
    const int assumed = old;
    old = atomicCAS(bits, assumed, __float_as_int(value));
    if (old == assumed) break;
  }
  return __int_as_float(old);
}

__global__ void image_extreme(const float* input, float* output, int n, bool maximum) {
  float value = maximum ? -1000.0f : 1000.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) value = maximum ? fmaxf(value, input[i]) : fminf(value, input[i]);
  for (int offset = 16; offset > 0; offset /= 2) {
    const float other = __shfl_down_sync(0xffffffff, value, offset);
    value = maximum ? fmaxf(value, other) : fminf(value, other);
  }
  if ((threadIdx.x & 31) == 0) {
    if (maximum) atomic_max_float(output, value); else atomic_min_float(output, value);
  }
}

__global__ void image_extend(float* x, const float* minimum, const float* maximum, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) {
    const float value = 5.0f * (x[i] - *minimum) / fmaxf(*maximum - *minimum, 1.0e-12f);
    x[i] = value > 1.0f ? 1.0f : value;
  }
}

__global__ void image_unsharpen(const float* x, const float* y, float* output, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) output[i] = fminf(1.0f, fmaxf(0.0f, 1.5f * x[i] - 0.5f * y[i]));
}

__global__ void image_combine(const float* x, const float* y, const float* mask,
                              float* output, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) output[i] = x[i] * mask[i] + y[i] * (1.0f - mask[i]);
}

__global__ void dl_conv2d(float* output, const float* x, const float* kernels,
                          int n, int channels, int kernel_width, int out_channels,
                          int stride) {
  extern __shared__ float local_kernel[];
  const int kernel_count = channels * kernel_width * kernel_width * out_channels;
  const int local = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = local; i < kernel_count; i += blockDim.x * blockDim.y) local_kernel[i] = kernels[i];
  __syncthreads();
  const int radius = kernel_width / 2;
  const int out_width = n / stride;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < out_width - radius; row += blockDim.x * gridDim.x) {
    for (int col = blockIdx.y * blockDim.y + threadIdx.y;
         col < out_width - radius; col += blockDim.y * gridDim.y) {
      for (int out_channel = 0; out_channel < out_channels; ++out_channel) {
        float value = 0.0f;
        for (int kr = -radius; kr <= radius; ++kr) for (int kc = -radius; kc <= radius; ++kc) {
          const int input_row = row * stride + radius + kr;
          const int input_col = col * stride + radius + kc;
          for (int channel = 0; channel < channels; ++channel) {
            const int kernel_index = channel + channels * ((kc + radius) + kernel_width * ((kr + radius) + kernel_width * out_channel));
            value += local_kernel[kernel_index] * x[(input_row * n + input_col) * channels + channel];
          }
        }
        output[out_channel + out_channels * (col + out_width * row)] = fmaxf(value, 0.0f);
      }
    }
  }
}

__global__ void dl_mean_pool(float* output, const float* x, int n, int channels,
                             int diameter) {
  const int radius = diameter / 2;
  const int out_width = n / diameter;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < out_width - radius; row += blockDim.x * gridDim.x) {
    for (int col = blockIdx.y * blockDim.y + threadIdx.y;
         col < out_width - radius; col += blockDim.y * gridDim.y) {
      for (int channel = blockIdx.z * blockDim.z + threadIdx.z;
           channel < channels; channel += blockDim.z * gridDim.z) {
        float value = 0.0f;
        for (int kr = -radius; kr <= radius; ++kr) for (int kc = -radius; kc <= radius; ++kc) {
          const int ir = row * diameter + radius + kr;
          const int ic = col * diameter + radius + kc;
          value += x[(ir * n + ic) * channels + channel];
        }
        output[channel + channels * (col + out_width * row)] = value / (diameter * diameter);
      }
    }
  }
}

__global__ void dl_concat(float* output, const float* x, const float* y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) { output[i] = x[i]; output[i + n] = y[i]; }
}

__global__ void dl_dot(const float* x, const float* y, float* output, int n) {
  float value = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) value += x[i] * y[i];
  value = warp_sum(value);
  if ((threadIdx.x & 31) == 0) atomicAdd(output, value);
}

struct Workspace {
  BenchmarkKind kind = BenchmarkKind::SYNTHETIC;
  std::vector<void*> allocations;
  std::array<float*, 16> f{};
  std::array<double*, 20> d{};
  std::array<int*, 8> i{};
  int n = 0;
  int aux = 0;
};

template <typename T>
inline T* allocate(Workspace& workspace, std::size_t count) {
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
  workspace.allocations.push_back(pointer);
  return pointer;
}

template <typename T>
inline void initialize(T* pointer, std::size_t count, T scale) {
  initialize_array<<<kBlocks, 256>>>(pointer, count, scale);
  CUDA_CHECK(cudaGetLastError());
}

inline void copy_gaussian(float* device, int diameter, float sigma) {
  std::vector<float> values(static_cast<std::size_t>(diameter * diameter));
  const int mean = diameter / 2;
  float sum = 0.0f;
  for (int row = 0; row < diameter; ++row) for (int col = 0; col < diameter; ++col) {
    const float value = std::exp(-0.5f * ((row - mean) * (row - mean) + (col - mean) * (col - mean)) / (sigma * sigma));
    values[static_cast<std::size_t>(row * diameter + col)] = value; sum += value;
  }
  for (float& value : values) value /= sum;
  CUDA_CHECK(cudaMemcpy(device, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
}

inline Workspace create_workspace(BenchmarkKind kind) {
  Workspace w;
  w.kind = kind;
  if (kind == BenchmarkKind::SYNTHETIC) return w;

  if (kind == BenchmarkKind::VECTOR_SQUARE) {
    w.n = kVectorElements;
    for (int j = 0; j < 4; ++j) w.f[j] = allocate<float>(w, w.n);
    w.f[4] = allocate<float>(w, 1);
    initialize(w.f[0], w.n, 0.001f); initialize(w.f[1], w.n, 0.002f);
    CUDA_CHECK(cudaMemset(w.f[4], 0, sizeof(float)));
  } else if (kind == BenchmarkKind::BLACK_SCHOLES) {
    w.n = kBlackScholesElements;
    for (int j = 0; j < 20; ++j) w.d[j] = allocate<double>(w, w.n);
    for (int j = 0; j < 10; ++j) initialize(w.d[j], w.n, 0.8);
  } else if (kind == BenchmarkKind::MACHINE_LEARNING) {
    constexpr int features = 200, classes = 10;
    w.n = kMlRows; w.aux = features;
    w.i[0] = allocate<int>(w, static_cast<std::size_t>(w.n) * features);
    w.i[1] = allocate<int>(w, w.n);
    w.f[0] = allocate<float>(w, static_cast<std::size_t>(w.n) * features);
    w.f[1] = allocate<float>(w, features * classes);
    w.f[2] = allocate<float>(w, features * classes);
    w.f[3] = allocate<float>(w, classes);
    w.f[4] = allocate<float>(w, w.n); w.f[5] = allocate<float>(w, w.n);
    w.f[6] = allocate<float>(w, static_cast<std::size_t>(w.n) * classes);
    w.f[7] = allocate<float>(w, static_cast<std::size_t>(w.n) * classes);
    initialize(w.i[0], static_cast<std::size_t>(w.n) * features, 1);
    initialize(w.f[1], features * classes, 0.0001f); initialize(w.f[2], features * classes, 0.0002f);
    initialize(w.f[3], classes, 0.001f);
    CUDA_CHECK(cudaMemset(w.f[6], 0, static_cast<std::size_t>(w.n) * classes * sizeof(float)));
    CUDA_CHECK(cudaMemset(w.f[7], 0, static_cast<std::size_t>(w.n) * classes * sizeof(float)));
  } else if (kind == BenchmarkKind::HITS) {
    constexpr int degree = 3;
    w.n = kHitsRows; w.aux = degree;
    for (int j = 0; j < 2; ++j) {
      w.i[j * 3] = allocate<int>(w, static_cast<std::size_t>(w.n) + 1);
      w.i[j * 3 + 1] = allocate<int>(w, static_cast<std::size_t>(w.n) * degree);
      w.i[j * 3 + 2] = allocate<int>(w, static_cast<std::size_t>(w.n) * degree);
      initialize_csr<<<kBlocks, 256>>>(w.i[j * 3], w.i[j * 3 + 1], w.i[j * 3 + 2], w.n, degree);
    }
    for (int j = 0; j < 4; ++j) w.f[j] = allocate<float>(w, w.n);
    w.f[4] = allocate<float>(w, 1); w.f[5] = allocate<float>(w, 1);
    initialize(w.f[0], w.n, 1.0f); initialize(w.f[2], w.n, 1.0f);
    CUDA_CHECK(cudaMemset(w.f[4], 0, sizeof(float))); CUDA_CHECK(cudaMemset(w.f[5], 0, sizeof(float)));
  } else if (kind == BenchmarkKind::IMAGE_PROCESSING) {
    w.n = kImageWidth; const std::size_t pixels = static_cast<std::size_t>(w.n) * w.n;
    for (int j = 0; j < 10; ++j) w.f[j] = allocate<float>(w, pixels);
    w.f[10] = allocate<float>(w, 9); w.f[11] = allocate<float>(w, 25); w.f[12] = allocate<float>(w, 9);
    w.f[13] = allocate<float>(w, 1); w.f[14] = allocate<float>(w, 1);
    initialize(w.f[0], pixels, 0.01f);
    copy_gaussian(w.f[10], 3, 1.0f); copy_gaussian(w.f[11], 5, 10.0f); copy_gaussian(w.f[12], 3, 5.0f);
    const float maximum = -1000.0f, minimum = 1000.0f;
    CUDA_CHECK(cudaMemcpy(w.f[13], &maximum, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.f[14], &minimum, sizeof(float), cudaMemcpyHostToDevice));
  } else if (kind == BenchmarkKind::DEEP_LEARNING) {
    constexpr int channels = 1, k = 3, c1 = 8, c2 = 16, stride = 2, pool = 5;
    w.n = kDlWidth;
    const std::size_t input = static_cast<std::size_t>(w.n) * w.n * channels;
    const std::size_t layer1 = static_cast<std::size_t>(w.n / stride) * (w.n / stride) * c1;
    const std::size_t pooled = layer1 / (pool * pool);
    const std::size_t layer2 = static_cast<std::size_t>(w.n / stride / pool / stride) * (w.n / stride / pool / stride) * c2;
    w.aux = static_cast<int>(layer2);
    w.f[0] = allocate<float>(w, input); w.f[1] = allocate<float>(w, input);
    w.f[2] = allocate<float>(w, layer1); w.f[3] = allocate<float>(w, layer1);
    w.f[4] = allocate<float>(w, pooled); w.f[5] = allocate<float>(w, pooled);
    w.f[6] = allocate<float>(w, layer2); w.f[7] = allocate<float>(w, layer2);
    w.f[8] = allocate<float>(w, channels * k * k * c1); w.f[9] = allocate<float>(w, c1 * k * k * c2);
    w.f[10] = allocate<float>(w, channels * k * k * c1); w.f[11] = allocate<float>(w, c1 * k * k * c2);
    w.f[12] = allocate<float>(w, layer2 * 2); w.f[13] = allocate<float>(w, layer2 * 2); w.f[14] = allocate<float>(w, 1);
    initialize(w.f[0], input, 0.01f); initialize(w.f[1], input, 0.02f);
    initialize(w.f[8], channels * k * k * c1, 0.001f); initialize(w.f[9], c1 * k * k * c2, 0.001f);
    initialize(w.f[10], channels * k * k * c1, 0.002f); initialize(w.f[11], c1 * k * k * c2, 0.002f);
    initialize(w.f[13], layer2 * 2, 0.0001f); CUDA_CHECK(cudaMemset(w.f[14], 0, sizeof(float)));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return w;
}

inline void destroy_workspace(Workspace& workspace) {
  for (void* pointer : workspace.allocations) CUDA_CHECK(cudaFree(pointer));
  workspace.allocations.clear();
}

inline void launch(const TaskSpec& task, Workspace& w, cudaStream_t stream) {
  switch (w.kind) {
    case BenchmarkKind::VECTOR_SQUARE:
      if (task.id == 0) vector_square<<<kBlocks,kThreads,0,stream>>>(w.f[0],w.f[2],w.n);
      else if (task.id == 1) vector_square<<<kBlocks,kThreads,0,stream>>>(w.f[1],w.f[3],w.n);
      else vector_reduce<<<kBlocks,kThreads,0,stream>>>(w.f[2],w.f[3],w.f[4],w.n);
      break;
    case BenchmarkKind::BLACK_SCHOLES:
      black_scholes<<<kBlocks,kBlackScholesThreads,0,stream>>>(w.d[task.id],w.d[task.id+10],w.n);
      break;
    case BenchmarkKind::MACHINE_LEARNING: {
      constexpr int f=200,c=10;
      switch(task.id) {
        case 0: ml_rr_scale<<<kBlocks,kThreads,0,stream>>>(w.i[0],w.f[0],w.n,f); break;
        case 1: ml_matmul<<<kBlocks,kThreads,0,stream>>>(w.i[0],w.f[1],w.f[6],w.n,f,c); break;
        case 2: ml_matmul<<<kBlocks,kThreads,0,stream>>>(w.f[0],w.f[2],w.f[7],w.n,f,c); break;
        case 3: ml_row_max<<<kBlocks,kThreads,0,stream>>>(w.f[6],w.f[4],w.n,c); break;
        case 4: ml_logsumexp<<<kBlocks,kThreads,0,stream>>>(w.f[6],w.f[4],w.f[5],w.n,c); break;
        case 5: ml_add_intercept<<<kBlocks,kThreads,0,stream>>>(w.f[7],w.f[3],w.n,c); break;
        case 6: ml_normalize<<<kBlocks,kThreads,0,stream>>>(w.f[6],w.f[5],w.n,c); break;
        case 7: ml_softmax<<<kBlocks,kThreads,0,stream>>>(w.f[6],w.n,c); break;
        case 8: ml_softmax<<<kBlocks,kThreads,0,stream>>>(w.f[7],w.n,c); break;
        default: ml_argmax<<<kBlocks,kThreads,0,stream>>>(w.f[6],w.f[7],w.i[1],w.n,c); break;
      } break;
    }
    case BenchmarkKind::HITS: {
      const int phase=task.id%7;
      if(phase==0) hits_spmv<<<kBlocks,kThreads,0,stream>>>(w.i[3],w.i[4],w.i[5],w.f[2],w.f[1],w.n);
      else if(phase==1) hits_spmv<<<kBlocks,kThreads,0,stream>>>(w.i[0],w.i[1],w.i[2],w.f[0],w.f[3],w.n);
      else if(phase==2) hits_sum<<<kBlocks,kThreads,0,stream>>>(w.f[1],w.f[4],w.n);
      else if(phase==3) hits_sum<<<kBlocks,kThreads,0,stream>>>(w.f[3],w.f[5],w.n);
      else if(phase==4) hits_divide<<<kBlocks,kThreads,0,stream>>>(w.f[1],w.f[0],w.f[4],w.n);
      else if(phase==5) hits_divide<<<kBlocks,kThreads,0,stream>>>(w.f[3],w.f[2],w.f[5],w.n);
      else hits_reset<<<1,1,0,stream>>>(w.f[4],w.f[5]);
      break;
    }
    case BenchmarkKind::IMAGE_PROCESSING: {
      const int pixels=w.n*w.n; dim3 threads(kImageThreads2d,kImageThreads2d); dim3 grid(8,8);
      if(task.id==0) image_blur<<<grid,threads,9*sizeof(float),stream>>>(w.f[0],w.f[4],w.n,w.f[10],3);
      else if(task.id==1) image_blur<<<grid,threads,25*sizeof(float),stream>>>(w.f[0],w.f[5],w.n,w.f[11],5);
      else if(task.id==2) image_blur<<<grid,threads,9*sizeof(float),stream>>>(w.f[0],w.f[6],w.n,w.f[12],3);
      else if(task.id==3) image_sobel<<<grid,threads,0,stream>>>(w.f[4],w.f[7],w.n);
      else if(task.id==4) image_sobel<<<grid,threads,0,stream>>>(w.f[5],w.f[8],w.n);
      else if(task.id==5) image_extreme<<<kBlocks,kThreads,0,stream>>>(w.f[8],w.f[13],pixels,true);
      else if(task.id==6) image_extreme<<<kBlocks,kThreads,0,stream>>>(w.f[8],w.f[14],pixels,false);
      else if(task.id==7) image_extend<<<kBlocks,kThreads,0,stream>>>(w.f[8],w.f[14],w.f[13],pixels);
      else if(task.id==8) image_unsharpen<<<kBlocks,kThreads,0,stream>>>(w.f[0],w.f[6],w.f[9],pixels);
      else if(task.id==9) image_combine<<<kBlocks,kThreads,0,stream>>>(w.f[9],w.f[5],w.f[8],w.f[1],pixels);
      else image_combine<<<kBlocks,kThreads,0,stream>>>(w.f[1],w.f[4],w.f[7],w.f[2],pixels);
      break;
    }
    case BenchmarkKind::DEEP_LEARNING: {
      constexpr int k=3,c1=8,c2=16,stride=2,pool=5; dim3 t(kDlThreads2d,kDlThreads2d); dim3 g(8,8); dim3 tp(4,4,4),gp(4,4,4);
      if(task.id==0) dl_conv2d<<<g,t,k*k*c1*sizeof(float),stream>>>(w.f[2],w.f[0],w.f[8],w.n,1,k,c1,stride);
      else if(task.id==1) dl_conv2d<<<g,t,k*k*c1*sizeof(float),stream>>>(w.f[3],w.f[1],w.f[10],w.n,1,k,c1,stride);
      else if(task.id==2) dl_mean_pool<<<gp,tp,0,stream>>>(w.f[4],w.f[2],w.n/stride,c1,pool);
      else if(task.id==3) dl_mean_pool<<<gp,tp,0,stream>>>(w.f[5],w.f[3],w.n/stride,c1,pool);
      else if(task.id==4) dl_conv2d<<<g,t,k*k*c1*c2*sizeof(float),stream>>>(w.f[6],w.f[4],w.f[9],w.n/stride/pool,c1,k,c2,stride);
      else if(task.id==5) dl_conv2d<<<g,t,k*k*c1*c2*sizeof(float),stream>>>(w.f[7],w.f[5],w.f[11],w.n/stride/pool,c1,k,c2,stride);
      else if(task.id==6) dl_concat<<<kBlocks,kThreads,0,stream>>>(w.f[12],w.f[6],w.f[7],w.aux);
      else dl_dot<<<kBlocks,kThreads,0,stream>>>(w.f[12],w.f[13],w.f[14],w.aux*2);
      break;
    }
    case BenchmarkKind::SYNTHETIC: throw std::logic_error("synthetic task sent to KESCO launcher");
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace stg::kesco
