#include "taichi_gfx_impl.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"

GfxRuntime::GfxRuntime(taichi::Arch arch) : Runtime(arch) {
}

Error GfxRuntime::create_aot_module(const taichi::io::VirtualDir *dir,
                                    TiAotModule &out) {
  taichi::lang::gfx::AotModuleParams params{};
  params.dir = dir;
  params.runtime = &get_gfx_runtime();
  std::unique_ptr<taichi::lang::aot::Module> aot_module =
      taichi::lang::aot::Module::load(arch, params);
  if (aot_module->is_corrupted()) {
    return Error(TI_ERROR_CORRUPTED_DATA, "aot_module");
  }

  const taichi::lang::DeviceCapabilityConfig &current_devcaps =
      params.runtime->get_ti_device()->get_caps();
  const taichi::lang::DeviceCapabilityConfig &required_devcaps =
      aot_module->get_required_caps();
  for (const auto &pair : required_devcaps.devcaps) {
    uint32_t current_version = current_devcaps.get(pair.first);
    uint32_t required_version = pair.second;
    if (current_version != required_version) {
      return Error(TI_ERROR_INCOMPATIBLE_MODULE,
                   taichi::lang::to_string(pair.first).c_str());
    }
  }

  size_t root_size = aot_module->get_root_size();
  params.runtime->add_root_buffer(root_size);
  out = (TiAotModule) new AotModule(*this, std::move(aot_module));
  return Error();
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
void GfxRuntime::track_image(const taichi::lang::DeviceAllocation &image,
                             taichi::lang::ImageLayout layout) {
  get_gfx_runtime().track_image(image, layout);
}
void GfxRuntime::untrack_image(const taichi::lang::DeviceAllocation &image) {
  get_gfx_runtime().untrack_image(image);
}
void GfxRuntime::transition_image(const taichi::lang::DeviceAllocation &image,
                                  taichi::lang::ImageLayout layout) {
  get_gfx_runtime().transition_image(image, layout);
}
void GfxRuntime::flush() {
  get_gfx_runtime().flush();
}
void GfxRuntime::wait() {
  // (penguinliong) It's currently waiting for the entire runtime to stop.
  // Should be simply waiting for its fence to finish.
  get_gfx_runtime().synchronize();
}
