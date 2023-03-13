#pragma once
#include "taichi/runtime/gfx/runtime.h"

namespace taichi::lang {
namespace gfx {
class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl(GfxRuntime *runtime, GfxRuntime::RegisterParams &&params)
      : runtime_(runtime), params_(std::move(params)) {
    handle_ = runtime_->register_taichi_kernel(params_);
    arch = Arch::vulkan;  // Only for letting the launch context builder know
                          // the arch does not use LLVM.
                          // TODO: remove arch after the refactoring of
                          //  SPIR-V based backends completes.
  }

  void launch(RuntimeContext *ctx) override {
    runtime_->launch_kernel(handle_, ctx);
  }

  const GfxRuntime::RegisterParams &params() {
    return params_;
  }

 private:
  GfxRuntime *const runtime_;
  GfxRuntime::KernelHandle handle_;
  const GfxRuntime::RegisterParams params_;
};
}  // namespace gfx
}  // namespace taichi::lang
