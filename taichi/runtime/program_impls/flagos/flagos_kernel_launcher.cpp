#include "taichi/runtime/program_impls/flagos/flagos_kernel_launcher.h"

#include "taichi/rhi/flagos/flagos_device.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/program/kernel.h"

namespace taichi {
namespace lang {
namespace flagos {

KernelLauncher::KernelLauncher(Config config) : config_(std::move(config)) {
}

void KernelLauncher::launch_kernel(
    LlvmLaunchArgBuilder &arg_builder,
    Kernel *kernel,
    const LLVM::CompiledKernelData &compiled_kernel) {
  // Get the FlagOS device
  auto *device =
      static_cast<FlagosDevice *>(config_.executor->get_compute_device());
  TI_ASSERT(device);

  // Get kernel data
  const auto &kernel_data = compiled_kernel.get_internal_data();
  const auto &task_funcs = kernel_data.tasks;
  const auto &args = kernel_data.args;

  // Prepare launch arguments
  std::vector<void *> arg_pointers;
  arg_pointers.reserve(args.size());

  for (const auto &arg : args) {
    arg_pointers.push_back(const_cast<void *>(arg.data));
  }

  // Launch each task (offloaded kernel)
  for (const auto &task : task_funcs) {
    const std::string &kernel_name = task.name;
    uint32_t grid_dim = task.grid_dim;
    uint32_t block_dim = task.block_dim;

    TI_TRACE("Launching FlagOS kernel: {}, grid_dim: {}, block_dim: {}",
             kernel_name, grid_dim, block_dim);

    // Launch kernel via FlagOS device
    // This will call FlagTree to compile and execute the kernel
    device->launch_kernel(kernel_name, arg_pointers.data(), grid_dim,
                          block_dim);
  }

  // Synchronize if needed
  if (kernel->program->compile_config().sync_after_kernel_launch) {
    device->wait_idle();
  }
}

}  // namespace flagos
}  // namespace lang
}  // namespace taichi
