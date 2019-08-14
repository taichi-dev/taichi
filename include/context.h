#pragma once
#include "common.h"
#include <string>

TLANG_NAMESPACE_BEGIN

class CPUProfiler;

struct Context {
  using Buffer = void *;
  Buffer buffers[1];
  uint64 args[max_num_args];

  void *leaves;
  int num_leaves;
  CPUProfiler *cpu_profiler;
  void *runtime;

  Context() {
    leaves = 0;
    num_leaves = 0;
    for (int i = 0; i < 1; i++)
      buffers[i] = nullptr;
  }

  Context(void *x) : Context() {
    buffers[0] = x;
  }

  template <typename T>
  __host__ __device__ T get_arg(int i) {
    return union_cast_different_size<T>(args[i]);
  }

  template <typename T>
  __host__ __device__ void set_arg(int i, T v) {
    args[i] = union_cast_different_size<uint64>(v);
  }
};

TLANG_NAMESPACE_END
