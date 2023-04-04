#pragma once

#include "taichi/program/kernel_launcher.h"
#include "taichi/runtime/gfx/runtime.h"

namespace taichi::lang {
namespace gfx {

class KernelLauncher : public lang::KernelLauncher {
 public:
  struct Config {
    GfxRuntime *gfx_runtime_{nullptr};
  };

  explicit KernelLauncher(Config config);

  void launch_kernel(const lang::CompiledKernelData &compiled_kernel_data,
                     LaunchContextBuilder &ctx) override;

 private:
  Handle register_kernel(const lang::CompiledKernelData &compiled_kernel_data);

  Config config_;
};

}  // namespace gfx
}  // namespace taichi::lang
