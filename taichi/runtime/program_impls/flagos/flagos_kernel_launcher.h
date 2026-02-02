#pragma once

#include "taichi/runtime/llvm/kernel_launcher.h"

namespace taichi {
namespace lang {
namespace flagos {

/**
 * @brief FlagOS Kernel Launcher
 *
 * This class handles kernel launching for FlagOS-supported AI chips.
 * It integrates with the FlagOS runtime to execute compiled kernels.
 */

class KernelLauncher : public LLVM::KernelLauncher {
 public:
  using Config = LLVM::KernelLauncher::Config;

  explicit KernelLauncher(Config config);

  /**
   * @brief Launch a kernel on FlagOS device
   */
  void launch_kernel(LlvmLaunchArgBuilder &arg_builder,
                     Kernel *kernel,
                     const LLVM::CompiledKernelData &compiled_kernel) override;

 private:
  Config config_;
};

}  // namespace flagos
}  // namespace lang
}  // namespace taichi
