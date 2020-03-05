#pragma once

// Use relative path here for runtime compilation
#include "taichi/inc/constants.h"

#if defined(TI_RUNTIME_HOST)
namespace taichi::lang {
using namespace taichi;
#endif

struct LLVMRuntime;

// "Context" holds necessary data for function calls, such as arguments and
// LLVMRuntime struct
struct Context {
  LLVMRuntime *runtime;
  uint64 args[taichi_max_num_args];
  int32 extra_args[taichi_max_num_args][taichi_max_num_indices];

  static constexpr size_t extra_args_size = sizeof(extra_args);

#if defined(TI_RUNTIME_HOST)
  template <typename T>
  T get_arg(int i) {
    return taichi_union_cast_with_different_sizes<T>(args[i]);
  }

  uint64 get_arg_as_uint64(int i) {
    return args[i];
  }

  template <typename T>
  void set_arg(int i, T v) {
    args[i] = taichi_union_cast_with_different_sizes<uint64>(v);
  }
#endif
};

struct MemRequest {
  std::size_t size;
  std::size_t alignment;
  uint8 *ptr;
  std::size_t __padding;
};

static_assert((sizeof(MemRequest) & (sizeof(MemRequest) - 1)) == 0);

struct MemRequestQueue {
  MemRequest requests[taichi_max_num_mem_requests];
  int tail;
  int processed;
};

#if defined(TI_RUNTIME_HOST)
}
#endif
