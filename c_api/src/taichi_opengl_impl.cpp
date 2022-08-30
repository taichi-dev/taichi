#include "taichi_opengl_impl.h"

#ifdef TI_WITH_OPENGL

OpenglRuntime::OpenglRuntime()
    : GfxRuntime(taichi::Arch::opengl),
      gfx_runtime_(taichi::lang::gfx::GfxRuntime::Params{
          host_result_buffer_.data(), &device_}) {
  device_.set_cap(taichi::lang::DeviceCapability::spirv_has_int64, true);
  device_.set_cap(taichi::lang::DeviceCapability::spirv_has_float64, true);
  device_.set_cap(taichi::lang::DeviceCapability::spirv_version, 0x10300);
}
taichi::lang::Device &OpenglRuntime::get() {
  return static_cast<taichi::lang::Device &>(device_);
}
taichi::lang::gfx::GfxRuntime &OpenglRuntime::get_gfx_runtime() {
  return gfx_runtime_;
}
#endif
