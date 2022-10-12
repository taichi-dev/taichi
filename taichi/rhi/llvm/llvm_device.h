#pragma once

#include "taichi/rhi/device.h"

namespace taichi::lang {

class LlvmDevice : public Device {
 public:
  struct LlvmRuntimeAllocParams : AllocParams {
    bool use_cached{true};
    JITModule *runtime_jit{nullptr};
    LLVMRuntime *runtime{nullptr};
    uint64 *result_buffer{nullptr};
  };

  virtual DeviceAllocation allocate_memory_runtime(
      const LlvmRuntimeAllocParams &params) {
    TI_NOT_IMPLEMENTED;
  }

  uint64_t *allocate_llvm_runtime_memory_jit(
      const LlvmRuntimeAllocParams &params);
};

}  // namespace taichi::lang
