#pragma once

#if !defined(TC_INCLUDED)

#include <cstdio>
#include <cstdlib>
#include <cstring>
using float32 = float;
using float64 = double;
using uint64 = unsigned long long;

#if defined(TLANG_CPU)
#include <immintrin.h>
#endif

#if defined(TLANG_GPU)
#include <cuda_runtime.h>
#endif

#endif

namespace taichi {
namespace Tlang {

struct Context {
  static constexpr int max_num_buffers = 16;
  static constexpr int max_num_parameters = 16;
  static constexpr int max_num_ranges = 16;
  using Buffer = void *;
  Buffer buffers[max_num_buffers];
  double parameters[max_num_parameters];
  uint64 ranges[max_num_ranges];

  Context() {
    std::memset(buffers, 0, sizeof(buffers));
    std::memset(parameters, 0, sizeof(parameters));
    std::memset(ranges, 0, sizeof(ranges));
  }

  Context(void *x, void *y, void *z, uint64 n) : Context() {
    buffers[0] = x;
    buffers[1] = y;
    buffers[2] = z;
    ranges[0] = n;
  }

  template <typename T>
  T *get_buffer(int i) {
    return reinterpret_cast<T *>(buffers[i]);
  }

  template <typename T>
  T &get_parameter(int i) {
    return *reinterpret_cast<T *>(&parameters[i]);
  }

  uint64 &get_range(int i) {
    return ranges[i];
  }
};
}
}
