#ifdef TI_WITH_METAL
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/rhi/metal/metal_device.h"
#include "taichi_metal_impl.h"

namespace capi {

MetalRuntime::MetalRuntime(taichi::Arch arch)
    : GfxRuntime(arch), mtl_device_(taichi::lang::metal::MetalDevice::create()),
      gfx_runtime_(taichi::lang::gfx::GfxRuntime::Params{
          host_result_buffer_.data(), mtl_device_.get()}) {}

taichi::lang::Device &MetalRuntime::get() {
  return static_cast<taichi::lang::Device &>(*mtl_device_);
}
taichi::lang::gfx::GfxRuntime &MetalRuntime::get_gfx_runtime() {
  return gfx_runtime_;
}

taichi::lang::metal::MetalDevice &MetalRuntime::get_mtl() {
  return *mtl_device_;
}

} // namespace capi

#endif // TI_WITH_METAL
