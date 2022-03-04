#pragma once

// Use relative path here for runtime compilation
#include "taichi/inc/constants.h"

#if defined(TI_RUNTIME_HOST)
namespace taichi {
namespace lang {
#endif

struct LLVMRuntime;

// "RuntimeContext" holds necessary data for kernel body execution, such as a
// pointer to the LLVMRuntime struct, kernel arguments, and the thread id (if on
// CPU).
struct RuntimeContext {
  LLVMRuntime *runtime;
  // args can contain:
  // - primitive_types
  // - raw ptrs: for external array, or torch-based ndarray
  // - DeviceAllocation*: for taichi ndaray
  uint64 args[taichi_max_num_args_total];
  int32 extra_args[taichi_max_num_args_extra][taichi_max_num_indices];
  int32 cpu_thread_id;
  // |is_device_allocation| is true iff args[i] is a DeviceAllocation*.
  bool is_device_allocation[taichi_max_num_args_total]{false};
  uint64 *result_buffer;

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

  void set_device_allocation(int i, bool is_device_allocation_) {
    is_device_allocation[i] = is_device_allocation_;
  }
#endif
};

#if defined(TI_RUNTIME_HOST)
}  // namespace lang
}  // namespace taichi
#endif
