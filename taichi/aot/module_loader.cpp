#include "taichi/aot/module_loader.h"

namespace taichi {
namespace lang {
namespace aot {

std::unique_ptr<Module> Module::load(const std::string &path,
                                     Arch arch,
                                     std::any mod_params) {
//  arch_ = arch;
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

//Arch Module::arch() const {
//  return arch_;
//}

}  // namespace aot
}  // namespace lang
}  // namespace taichi
