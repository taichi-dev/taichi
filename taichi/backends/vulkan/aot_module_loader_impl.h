#pragma once

#include <any>
#include <string>
#include <vector>

#include "taichi/backends/vulkan/aot_utils.h"
#include "taichi/runtime/vulkan/runtime.h"
#include "taichi/codegen/spirv/kernel_utils.h"

#include "taichi/aot/module_loader.h"

namespace taichi {
namespace lang {
namespace vulkan {

class VkRuntime;

class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl(VkRuntime *runtime, VkRuntime::RegisterParams &&params)
      : runtime_(runtime), params_(std::move(params)) {
  }

  void launch(RuntimeContext *ctx) override {
    auto handle = runtime_->register_taichi_kernel(params_);
    runtime_->launch_kernel(handle, ctx);
  }

 private:
  VkRuntime *const runtime_;
  const VkRuntime::RegisterParams params_;
};
struct TI_DLL_EXPORT AotModuleParams {
  std::string module_path;
  VkRuntime *runtime{nullptr};
};

TI_DLL_EXPORT std::unique_ptr<aot::Module> make_aot_module(std::any mod_params);

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
