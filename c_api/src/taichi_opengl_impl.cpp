#ifdef TI_WITH_OPENGL
#include "taichi_opengl_impl.h"

OpenglRuntime::OpenglRuntime()
    : GfxRuntime(taichi::Arch::opengl),
      device_(),
      gfx_runtime_(taichi::lang::gfx::GfxRuntime::Params{&device_}) {
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

TiRuntime ti_import_opengl_runtime(TiOpenglRuntimeInteropInfo *interop_info,
                                   bool use_gles) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(interop_info);
  taichi::lang::opengl::imported_process_address = interop_info->get_proc_addr;
  taichi::lang::opengl::set_gles_override(use_gles);
  TI_CAPI_TRY_CATCH_END();
  return ti_create_runtime(TI_ARCH_OPENGL, 0);
}

void ti_export_opengl_runtime(TiRuntime runtime,
                              TiOpenglRuntimeInteropInfo *interop_info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  // FIXME: (penguinliogn)
  interop_info->get_proc_addr =
      taichi::lang::opengl::kGetOpenglProcAddr.value();
  TI_CAPI_TRY_CATCH_END();
}

TiMemory ti_import_opengl_memory(TiRuntime runtime,
                                 TiOpenglMemoryInteropInfo *interop_info) {
  TiMemory out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(interop_info);

  OpenglRuntime *runtime2 = static_cast<OpenglRuntime *>((Runtime *)runtime);
  taichi::lang::DeviceAllocation devalloc{};
  devalloc.device = &runtime2->get_gl();
  devalloc.alloc_id = (taichi::lang::DeviceAllocationId)interop_info->buffer;
  out = devalloc2devmem(*runtime2, devalloc);
  TI_CAPI_TRY_CATCH_END();
  return out;
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
  interop_info->size =
      (GLsizeiptr)runtime2->get_gl().get_devalloc_size(devalloc);
  TI_CAPI_TRY_CATCH_END();
}

TiImage ti_import_opengl_image(TiRuntime runtime,
                               TiOpenglImageInteropInfo *interop_info) {
  TiImage out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(interop_info);
  TI_CAPI_INVALID_INTEROP_ARCH_RV(((Runtime *)runtime)->arch, opengl);

  OpenglRuntime *runtime2 = static_cast<OpenglRuntime *>((Runtime *)runtime);
  taichi::lang::opengl::GLImageAllocation gl_image{};
  gl_image.target = interop_info->target;
  gl_image.levels = interop_info->levels;
  gl_image.format = interop_info->format;
  gl_image.width = interop_info->width;
  gl_image.height = interop_info->height;
  gl_image.depth = interop_info->depth;
  gl_image.external = true;
  taichi::lang::DeviceAllocation devalloc = runtime2->get_gl().import_image(
      interop_info->texture, std::move(gl_image));
  out = devalloc2devimg(*runtime2, devalloc);
  TI_CAPI_TRY_CATCH_END();
  return out;
}

void ti_export_opengl_image(TiRuntime runtime,
                            TiImage image,
                            TiOpenglImageInteropInfo *interop_info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(image);
  TI_CAPI_ARGUMENT_NULL(interop_info);
  TI_CAPI_INVALID_INTEROP_ARCH(((Runtime *)runtime)->arch, opengl);

  OpenglRuntime *runtime2 = static_cast<OpenglRuntime *>((Runtime *)runtime);
  taichi::lang::DeviceAllocation devalloc = devimg2devalloc(*runtime2, image);
  GLuint texture = (GLuint)devalloc.alloc_id;
  const taichi::lang::opengl::GLImageAllocation &gl_image =
      runtime2->get_gl().get_gl_image(texture);
  interop_info->texture = texture;
  interop_info->target = gl_image.target;
  interop_info->levels = gl_image.levels;
  interop_info->format = gl_image.format;
  interop_info->width = gl_image.width;
  interop_info->height = gl_image.height;
  interop_info->depth = gl_image.depth;
  TI_CAPI_TRY_CATCH_END();
}

#endif  // TI_WITH_OPENGL
