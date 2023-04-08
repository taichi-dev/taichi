#include "taichi/runtime/llvm/kernel_launcher.h"

namespace taichi::lang {
namespace LLVM {

KernelLauncher::KernelLauncher(Config config) : config_(std::move(config)) {
}

void KernelLauncher::launch_kernel(
    const lang::CompiledKernelData &compiled_kernel_data,
    LaunchContextBuilder &ctx) {
  TI_ASSERT(arch_uses_llvm(compiled_kernel_data.arch()));
  const auto &llvm_ckd =
      dynamic_cast<const LLVM::CompiledKernelData &>(compiled_kernel_data);
  auto handle = register_llvm_kernel(llvm_ckd);
  launch_llvm_kernel(handle, ctx);
}

}  // namespace LLVM
}  // namespace taichi::lang
