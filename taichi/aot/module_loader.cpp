#include "taichi/aot/module_loader.h"

#include "taichi/backends/vulkan/aot_module_loader_impl.h"
#include "taichi/backends/metal/aot_module_loader_impl.h"

namespace taichi {
namespace lang {
namespace aot {

std::unique_ptr<Module> Module::load(const std::string &path,
                                     Arch arch,
                                     std::any mod_params) {
  if (arch == Arch::vulkan) {
#ifdef TI_WITH_VULKAN
    return vulkan::make_aot_module(mod_params);
#endif
  } else if (arch == Arch::metal) {
#ifdef TI_WITH_METAL
    return metal::make_aot_module(mod_params);
#endif
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

Kernel* Module::get_kernel(const std::string &name) {
  auto itr = loaded_kernels_.find(name);
  if (itr != loaded_kernels_.end()) {
    return itr->second.get();
  }
  auto k = make_new_kernel(name);
  auto *kptr = k.get();
  loaded_kernels_[name] = std::move(k);
  return kptr;
}

Field *Module::get_field(const std::string &name) {
  // TODO: Implement this
  TI_NOT_IMPLEMENTED;
  return nullptr;
}

}  // namespace aot
}  // namespace lang
}  // namespace taichi
