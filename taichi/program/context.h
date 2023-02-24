#pragma once

// Use relative path here for runtime compilation
#include "taichi/inc/constants.h"

#if defined(TI_RUNTIME_HOST)
#include "taichi/ir/type.h"
namespace taichi::lang {
#endif

struct LLVMRuntime;
struct DeviceAllocation;
class StructType;
// "RuntimeContext" holds necessary data for kernel body execution, such as a
// pointer to the LLVMRuntime struct, kernel arguments, and the thread id (if on
// CPU).
struct RuntimeContext {
  char *arg_buffer;
  size_t arg_buffer_size;
  const StructType *args_type;
  enum class DevAllocType : int8_t {
    kNone = 0,
    kNdarray = 1,
    kTexture = 2,
    kRWTexture = 3
  };

  LLVMRuntime *runtime{nullptr};
  // args can contain:
  // - primitive_types
  // - raw ptrs: for external array, or torch-based ndarray
  // - DeviceAllocation*: for taichi ndaray
  uint64 args[taichi_max_num_args_total];
  uint64 grad_args[taichi_max_num_args_total];
  int32 extra_args[taichi_max_num_args_extra][taichi_max_num_indices];
  int32 cpu_thread_id;

  bool has_grad[taichi_max_num_args_total];

  // Note that I've tried to group `array_runtime_size` and
  // `is_device_allocations` into a small struct. However, it caused some test
  // cases to stuck.

  // `array_runtime_size` records the runtime size of the
  // corresponding array arguments.
  uint64 array_runtime_sizes[taichi_max_num_args_total]{0};
  // `device_allocation_type` is set iff i-th arg is a `DeviceAllocation*`,
  // otherwise it is set to DevAllocType::kNone
  DevAllocType device_allocation_type[taichi_max_num_args_total]{
      DevAllocType::kNone};
  // We move the pointer of result buffer from LLVMRuntime to RuntimeContext
  // because each real function need a place to store its result, but
  // LLVMRuntime is shared among functions. So we moved the pointer to
  // RuntimeContext which each function have one.
  uint64 *result_buffer;
  size_t result_buffer_size{0};

  static constexpr size_t extra_args_size = sizeof(extra_args);

#if defined(TI_RUNTIME_HOST)
  template <typename T>
  T get_arg(int i) {
    if (arg_buffer_size > 0) {
      // Currently arg_buffer_size is always zero on non-LLVM-based backends
      return get_struct_arg<T>({i});
    }
    return taichi_union_cast_with_different_sizes<T>(args[i]);
  }

  template <typename T>
  T get_struct_arg(std::vector<int> index) {
    int offset = args_type->get_element_offset(index);
    return *(T *)(arg_buffer + offset);
  }
  template <typename T>
  T get_grad_arg(int i) {
    return taichi_union_cast_with_different_sizes<T>(grad_args[i]);
  }

  uint64 get_arg_as_uint64(int i) {
    return args[i];
  }

  template <typename T>
  void set_arg(int i, T v) {
    set_struct_arg({i}, v);
    args[i] = taichi_union_cast_with_different_sizes<uint64>(v);
    set_array_device_allocation_type(i, DevAllocType::kNone);
  }

  template <typename T>
  void set_struct_arg(std::vector<int> index, T v) {
    if (arg_buffer_size == 0) {
      // Currently arg_buffer_size is always zero on non-LLVM-based backends,
      // and this function is no-op for these backends.
      return;
    }
    int offset = args_type->get_element_offset(index);
    *(T *)(arg_buffer + offset) = v;
  }

  template <typename T>
  void set_grad_arg(int i, T v) {
    grad_args[i] = taichi_union_cast_with_different_sizes<uint64>(v);
  }

  void set_array_runtime_size(int i, uint64 size) {
    this->array_runtime_sizes[i] = size;
  }

  void set_array_device_allocation_type(int i, DevAllocType usage) {
    this->device_allocation_type[i] = usage;
  }

  template <typename T>
  T get_ret(int i) {
    return taichi_union_cast_with_different_sizes<T>(result_buffer[i]);
  }

  void set_arg_texture(int arg_id, intptr_t alloc_ptr) {
    set_struct_arg({arg_id}, alloc_ptr);
    args[arg_id] = taichi_union_cast_with_different_sizes<uint64>(alloc_ptr);
    set_array_device_allocation_type(arg_id, DevAllocType::kTexture);
  }

  void set_arg_rw_texture(int arg_id,
                          intptr_t alloc_ptr,
                          const std::array<int, 3> &shape) {
    set_struct_arg({arg_id}, alloc_ptr);
    args[arg_id] = taichi_union_cast_with_different_sizes<uint64>(alloc_ptr);
    set_array_device_allocation_type(arg_id, DevAllocType::kRWTexture);
    TI_ASSERT(shape.size() <= taichi_max_num_indices);
    for (int i = 0; i < shape.size(); i++) {
      extra_args[arg_id][i] = shape[i];
    }
  }

  void set_arg_external_array(int arg_id,
                              uintptr_t ptr,
                              uint64 size,
                              const std::vector<int64> &shape) {
    set_arg(arg_id, ptr);
    set_array_runtime_size(arg_id, size);
    set_array_device_allocation_type(arg_id,
                                     RuntimeContext::DevAllocType::kNone);
    for (uint64 i = 0; i < shape.size(); ++i) {
      extra_args[arg_id][i] = shape[i];
    }
  }

  void set_arg_ndarray(int arg_id,
                       intptr_t devalloc_ptr,
                       const std::vector<int> &shape,
                       bool has_grad = false,
                       intptr_t devalloc_ptr_grad = 0) {
    // Set has_grad value
    this->has_grad[arg_id] = has_grad;

    set_struct_arg({arg_id}, devalloc_ptr);
    // Set args[arg_id] value
    args[arg_id] = taichi_union_cast_with_different_sizes<uint64>(devalloc_ptr);

    // Set grad_args[arg_id] value
    if (has_grad) {
      grad_args[arg_id] =
          taichi_union_cast_with_different_sizes<uint64>(devalloc_ptr_grad);
    }

    // Set device allocation type and runtime size
    set_array_device_allocation_type(arg_id, DevAllocType::kNdarray);
    TI_ASSERT(shape.size() <= taichi_max_num_indices);
    size_t total_size = 1;
    for (int i = 0; i < shape.size(); i++) {
      extra_args[arg_id][i] = shape[i];
      total_size *= shape[i];
    }
    set_array_runtime_size(arg_id, total_size);
  }
#endif
};

#if defined(TI_RUNTIME_HOST)
}  // namespace taichi::lang
#endif
