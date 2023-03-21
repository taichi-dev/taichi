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
  char *arg_buffer{nullptr};
  size_t arg_buffer_size{0};
  const StructType *args_type{nullptr};
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
};

#if defined(TI_RUNTIME_HOST)
}  // namespace taichi::lang
#endif
