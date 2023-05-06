#pragma once

#include "taichi/rhi/device.h"

namespace taichi::lang {

class JITModule;
struct LLVMRuntime;

class LlvmDevice : public Device {
 public:
  struct LlvmRuntimeAllocParams : AllocParams {
    JITModule *runtime_jit{nullptr};
    LLVMRuntime *runtime{nullptr};
    uint64 *result_buffer{nullptr};
  };

  Arch arch() const override {
    TI_NOT_IMPLEMENTED
  }

  template <typename DEVICE>
  DEVICE *as() {
    auto *device = dynamic_cast<DEVICE *>(this);
    TI_ASSERT(device != nullptr);
    return device;
  }

  virtual DeviceAllocation import_memory(void *ptr, size_t size) {
    TI_NOT_IMPLEMENTED
  }

  virtual DeviceAllocation allocate_memory_runtime(
      const LlvmRuntimeAllocParams &params) {
    TI_NOT_IMPLEMENTED;
  }

  virtual void clear() {
    TI_NOT_IMPLEMENTED;
  }

  virtual uint64_t *allocate_llvm_runtime_memory_jit(
      const LlvmRuntimeAllocParams &params) {
    TI_NOT_IMPLEMENTED;
  }
};

}  // namespace taichi::lang
