#ifdef TI_WITH_OPENGL
#include "taichi_opengl_impl.h"

OpenglRuntime::OpenglRuntime()
    : GfxRuntime(taichi::Arch::opengl),
      device_(),
      gfx_runtime_(taichi::lang::gfx::GfxRuntime::Params{
          host_result_buffer_.data(), &device_}) {
  taichi::lang::DeviceCapabilityConfig caps{};
  caps.set(taichi::lang::DeviceCapability::spirv_has_int64, true);
  caps.set(taichi::lang::DeviceCapability::spirv_has_float64, true);
  caps.set(taichi::lang::DeviceCapability::spirv_version, 0x10300);
  get_gl().set_caps(std::move(caps));
}
taichi::lang::Device &OpenglRuntime::get() {
  return static_cast<taichi::lang::Device &>(device_);
}
taichi::lang::gfx::GfxRuntime &OpenglRuntime::get_gfx_runtime() {
  return gfx_runtime_;
}

void ti_export_opengl_runtime(TiRuntime runtime,
                              TiOpenglRuntimeInteropInfo *interop_info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  // FIXME: (penguinliogn)
  interop_info->get_proc_addr = taichi::lang::opengl::kGetOpenglProcAddr;
  TI_CAPI_TRY_CATCH_END();
}

void ti_export_opengl_memory(TiRuntime runtime,
                             TiMemory memory,
                             TiOpenglMemoryInteropInfo *interop_info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(memory);
  TI_CAPI_ARGUMENT_NULL(interop_info);
  TI_CAPI_INVALID_INTEROP_ARCH(((Runtime *)runtime)->arch, opengl);

  OpenglRuntime *runtime2 = static_cast<OpenglRuntime *>((Runtime *)runtime);
  taichi::lang::DeviceAllocation devalloc = devmem2devalloc(*runtime2, memory);
  interop_info->buffer = devalloc.alloc_id;
  interop_info->size = runtime2->get_gl().get_devalloc_size(devalloc);
  TI_CAPI_TRY_CATCH_END();
}

#endif  // TI_WITH_OPENGL
