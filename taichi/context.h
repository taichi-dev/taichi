#pragma once
#include "common.h"
#include <string>

TLANG_NAMESPACE_BEGIN

class CPUProfiler;

struct Context {
  void *root;
  uint64 args[max_num_args];
  int32 extra_args[max_num_args][max_num_indices];

  CPUProfiler *cpu_profiler;
  void *runtime;

  Context() {
    root = nullptr;
  }

  Context(void *x) : Context() {
    root = x;
  }

  template <typename T>
  T get_arg(int i) {
    return union_cast_different_size<T>(args[i]);
  }

  template <typename T>
  void set_arg(int i, T v) {
    args[i] = union_cast_different_size<uint64>(v);
  }
};

TLANG_NAMESPACE_END
