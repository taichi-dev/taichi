#include "taichi/runtime/runtime.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"

TLANG_NAMESPACE_BEGIN

#if !defined(TI_WITH_AMDGPU)
static_assert(
    false,
    "This file should not be compiled when TI_WITH_AMDGPU is undefined");
#endif

class RuntimeAMDGPU : public Runtime {
 public:
  RuntimeAMDGPU() {
  }

  std::size_t get_total_memory() override {
    return AMDGPUContext::get_instance().get_total_memory();
  }

  std::size_t get_available_memory() override {
    return AMDGPUContext::get_instance().get_free_memory();
  }

  bool detected() override {
    return AMDGPUContext::get_instance().detected();
  }

  ~RuntimeAMDGPU() override {
  }
};

static class RuntimeAMDGPUInjector {
 public:
  RuntimeAMDGPUInjector() {
    Runtime::register_impl<RuntimeAMDGPU>(Arch::amdgpu);
  }
} injector;

TLANG_NAMESPACE_END
