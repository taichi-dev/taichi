#include "taichi_metal_impl.h"
#include "runtime/metal/kernel_manager.h"

MetalRuntime::MetalRuntime() : GfxRuntime(taichi::Arch::metal) {
}

taichi::lang::metal::KernelManager::Params make_metal_compute_device_params(
  const taichi::lang::metal::CompiledRuntimeModule& runtime_module,
  uint64_t* host_result_buffer
) {
  taichi::lang::metal::KernelManager::Params params;
  params.compiled_runtime_module = runtime_module;
  params.config = &taichi::lang::default_compile_config;
  params.host_result_buffer = host_result_buffer;
  params.mem_pool = nullptr;
  params.profiler = nullptr;
  return params;
}
MetalRuntimeOwned::MetalRuntimeOwned() :
  runtime_module_(std::make_unique<taichi::lang::metal::CompiledRuntimeModule>(taichi::lang::metal::compile_runtime_module())),
  kernel_manager_(std::make_unique<taichi::lang::metal::KernelManager>(make_metal_compute_device_params(*runtime_module_, host_result_buffer_.data()))),
  device_(kernel_manager_->device().device),
  gfx_runtime_(taichi::lang::gfx::GfxRuntime::Params{ host_result_buffer_.data(), device_.get()}) {
}

taichi::lang::Device &MetalRuntimeOwned::get() {
  return *device_;
}
taichi::lang::gfx::GfxRuntime &MetalRuntimeOwned::get_gfx_runtime() {
  return gfx_runtime_;
}
