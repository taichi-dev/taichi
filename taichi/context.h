#pragma once

#include "constants.h"

#if defined(TI_RUNTIME_HOST)
#include "common.h"

namespace taichi::Tlang {
using namespace taichi;
#endif

struct Runtime;

// "Context" holds necessary data for function calls, such as arguments and
// Runtime struct
struct Context {
  Runtime *runtime;
  uint64 args[taichi_max_num_args];
  int32 extra_args[taichi_max_num_args][taichi_max_num_indices];

#if defined(TI_RUNTIME_HOST)
  template <typename T, typename G>
  static T union_cast_with_different_sizes(G g) {
    union {
      T t;
      G g;
    } u;
    u.g = g;
    return u.t;
  }

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

struct MemRequest {
  std::size_t size;
  std::size_t alignment;
  uint8 *ptr;
  std::size_t __padding;
};

struct MemRequestQueue {
  MemRequest requests[taichi_max_num_mem_requests];
  int tail;
  int processed;
};

#if defined(TI_RUNTIME_HOST)
}
#endif
