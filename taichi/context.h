#pragma once

#if defined(TI_RUNTIME_HOST)
#include "common.h"

namespace taichi::Tlang {
using namespace taichi;

template <typename T, typename G>
T union_cast_with_different_sizes(G g) {
  union {
    T t;
    G g;
  } u;
  u.g = g;
  return u.t;
}
#else
extern "C" {
#endif

struct Context {
  void *root;
  uint64 args[max_num_args];
  int32 extra_args[max_num_args][max_num_indices];

  void *cpu_profiler;
  void *runtime;

  Context() {
    root = nullptr;
  }

  Context(void *x) : Context() {
    root = x;
  }

#if defined(TI_RUNTIME_HOST)
  template <typename T>
  T get_arg(int i) {
    return union_cast_with_different_sizes<T>(args[i]);
  }

  template <typename T>
  void set_arg(int i, T v) {
    args[i] = union_cast_with_different_sizes<uint64>(v);
  }
#endif
};
}
