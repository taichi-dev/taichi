#pragma once

// Use relative path here for runtime compilation
#include "taichi/inc/constants.h"

#if defined(TI_RUNTIME_HOST)
namespace taichi {
namespace lang {
#endif

struct LLVMRuntime;
struct DeviceAllocation;
// "RuntimeContext" holds necessary data for kernel body execution, such as a
// pointer to the LLVMRuntime struct, kernel arguments, and the thread id (if on
// CPU).
struct RuntimeContext {
  LLVMRuntime *runtime{nullptr};
  // args can contain:
  // - primitive_types
  // - raw ptrs: for external array, or torch-based ndarray
  // - DeviceAllocation*: for taichi ndaray
  uint64 args[taichi_max_num_args_total];
  int32 extra_args[taichi_max_num_args_extra][taichi_max_num_indices];
  int32 cpu_thread_id;

  // Note that I've tried to group `array_runtime_size` and
  // `is_device_allocations` into a small struct. However, it caused some test
  // cases to stuck.

  // `array_runtime_size` records the runtime size of the
  // corresponding array arguments.
  uint64 array_runtime_sizes[taichi_max_num_args_total]{0};
  // `is_device_allocations` is true iff i-th arg is a `DeviceAllocation*`.
  bool is_device_allocations[taichi_max_num_args_total]{false};
  // We move the pointer of result buffer from LLVMRuntime to RuntimeContext
  // because each real function need a place to store its result, but
  // LLVMRuntime is shared among functions. So we moved the pointer to
  // RuntimeContext which each function have one.
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
    set_array_is_device_allocation(i, /*is_device_allocation=*/false);
  }

  void set_array_runtime_size(int i, uint64 size) {
    this->array_runtime_sizes[i] = size;
  }

  void set_array_is_device_allocation(int i, bool is_device_allocation) {
    this->is_device_allocations[i] = is_device_allocation;
  }

  template <typename T>
  T get_ret(int i) {
    return taichi_union_cast_with_different_sizes<T>(result_buffer[i]);
  }

  void set_arg_devalloc(int arg_id,
                        DeviceAllocation &alloc,
                        const std::vector<int> &shape) {
    args[arg_id] = taichi_union_cast_with_different_sizes<uint64>(&alloc);
    set_array_is_device_allocation(arg_id, /*is_device_allocation=*/true);
    TI_ASSERT(shape.size() <= taichi_max_num_indices);
    for (int i = 0; i < shape.size(); i++) {
      extra_args[arg_id][i] = shape[i];
    }
  }

  void set_arg_devalloc(int arg_id,
                        DeviceAllocation &alloc,
                        const std::vector<int> &shape,
                        const std::vector<int> &element_shape) {
    args[arg_id] = taichi_union_cast_with_different_sizes<uint64>(&alloc);
    set_array_is_device_allocation(arg_id, /*is_device_allocation=*/true);
    TI_ASSERT(shape.size() + element_shape.size() <= taichi_max_num_indices);
    for (int i = 0; i < shape.size(); i++) {
      extra_args[arg_id][i] = shape[i];
    }
    for (int i = 0; i < element_shape.size(); i++) {
      extra_args[arg_id][i + shape.size()] = element_shape[i];
    }
  }

#endif
};

#if defined(TI_RUNTIME_HOST)
}  // namespace lang
}  // namespace taichi
#endif
