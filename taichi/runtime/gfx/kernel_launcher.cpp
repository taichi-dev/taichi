#include "taichi/runtime/gfx/kernel_launcher.h"
#include "taichi/codegen/spirv/compiled_kernel_data.h"

namespace taichi::lang {
namespace gfx {

KernelLauncher::KernelLauncher(Config config) : config_(std::move(config)) {
}

void KernelLauncher::launch_kernel(
    const lang::CompiledKernelData &compiled_kernel_data,
    LaunchContextBuilder &ctx) {
  auto handle = register_kernel(compiled_kernel_data);
  config_.gfx_runtime_->launch_kernel(handle, ctx);
}

KernelLauncher::Handle KernelLauncher::register_kernel(
    const lang::CompiledKernelData &compiled_kernel_data) {
  if (!compiled_kernel_data.get_handle()) {
    const auto *spirv_compiled =
        dynamic_cast<const spirv::CompiledKernelData *>(&compiled_kernel_data);
    const auto &spirv_data = spirv_compiled->get_internal_data();
    gfx::GfxRuntime::RegisterParams params;
    params.kernel_attribs = spirv_data.metadata.kernel_attribs;
    params.task_spirv_source_codes = spirv_data.src.spirv_src;
    params.num_snode_trees = spirv_data.metadata.num_snode_trees;
    auto h = config_.gfx_runtime_->register_taichi_kernel(std::move(params));
    compiled_kernel_data.set_handle(h);
  }
  return *compiled_kernel_data.get_handle();
}

}  // namespace gfx
}  // namespace taichi::lang
