#pragma once
#include "taichi/runtime/gfx/runtime.h"

namespace taichi::lang {
namespace gfx {
class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl(GfxRuntime *runtime, GfxRuntime::RegisterParams &&params)
      : runtime_(runtime), params_(std::move(params)) {
    handle_ = runtime_->register_taichi_kernel(params_);
    ret_type = params_.kernel_attribs.ctx_attribs.rets_type();
    ret_size = params_.kernel_attribs.ctx_attribs.rets_bytes();
    args_type = params_.kernel_attribs.ctx_attribs.args_type();
    args_size = params_.kernel_attribs.ctx_attribs.args_bytes();
  }

  void launch(LaunchContextBuilder &ctx) override {
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
