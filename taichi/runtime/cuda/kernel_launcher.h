#pragma once

#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/kernel_launcher.h"

namespace taichi::lang {
namespace cuda {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  struct Context {
    JITModule *jit_module{nullptr};
    std::vector<Callable::Parameter> parameters;
    std::vector<OffloadedTask> offloaded_tasks;
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(const LLVM::CompiledKernelData &compiled) override;

 private:
  std::vector<Context> contexts_;
};

}  // namespace cuda
}  // namespace taichi::lang
