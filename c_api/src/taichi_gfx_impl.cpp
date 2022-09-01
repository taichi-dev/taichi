#include "taichi_gfx_impl.h"

GfxRuntime::GfxRuntime(taichi::Arch arch) : Runtime(arch) {
}

TiAotModule GfxRuntime::load_aot_module(const char *module_path) {
  taichi::lang::gfx::AotModuleParams params{};
  params.module_path = module_path;
  params.runtime = &get_gfx_runtime();
  std::unique_ptr<taichi::lang::aot::Module> aot_module =
      taichi::lang::aot::Module::load(arch, params);
  if (aot_module->is_corrupted()) {
    return TI_NULL_HANDLE;
  }
  size_t root_size = aot_module->get_root_size();
  params.runtime->add_root_buffer(root_size);
  return (TiAotModule)(new AotModule(*this, std::move(aot_module)));
}
void GfxRuntime::buffer_copy(const taichi::lang::DevicePtr &dst,
                             const taichi::lang::DevicePtr &src,
                             size_t size) {
  get_gfx_runtime().buffer_copy(dst, src, size);
}
void GfxRuntime::copy_image(const taichi::lang::DeviceAllocation &dst,
                            const taichi::lang::DeviceAllocation &src,
                            const taichi::lang::ImageCopyParams &params) {
  get_gfx_runtime().copy_image(dst, src, params);
}
void GfxRuntime::transition_image(const taichi::lang::DeviceAllocation &image,
                                  taichi::lang::ImageLayout layout) {
  get_gfx_runtime().transition_image(image, layout);
}
void GfxRuntime::submit() {
  get_gfx_runtime().flush();
}
void GfxRuntime::signal_event(taichi::lang::DeviceEvent *event) {
  get_gfx_runtime().signal_event(event);
}
void GfxRuntime::reset_event(taichi::lang::DeviceEvent *event) {
  get_gfx_runtime().reset_event(event);
}
void GfxRuntime::wait_event(taichi::lang::DeviceEvent *event) {
  get_gfx_runtime().wait_event(event);
}
void GfxRuntime::wait() {
  // (penguinliong) It's currently waiting for the entire runtime to stop.
  // Should be simply waiting for its fence to finish.
  get_gfx_runtime().synchronize();
}
