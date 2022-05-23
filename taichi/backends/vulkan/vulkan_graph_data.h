#pragma once
#include "taichi/runtime/vulkan/runtime.h"

namespace taichi {
namespace lang {
namespace vulkan {
class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl(VkRuntime *runtime, VkRuntime::RegisterParams &&params)
      : runtime_(runtime), params_(std::move(params)) {
    handle_ = runtime_->register_taichi_kernel(params_);
  }

  void launch(RuntimeContext *ctx) override {
    runtime_->launch_kernel(handle_, ctx);
  }

  const VkRuntime::RegisterParams &params() {
    return params_;
  }

 private:
  VkRuntime *const runtime_;
  VkRuntime::KernelHandle handle_;
  const VkRuntime::RegisterParams params_;
};
}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
