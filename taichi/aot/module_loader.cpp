#include "taichi/aot/module_loader.h"

#ifdef TI_WITH_VULKAN
#include "taichi/backends/vulkan/aot_module_loader_impl.h"
#endif
#ifdef TI_WITH_METAL
#include "taichi/backends/metal/aot_module_loader_impl.h"
#endif

namespace taichi {
namespace lang {
namespace aot {

std::unique_ptr<Module> Module::load(const std::string &path,
                                     Arch arch,
                                     std::any mod_params) {
  if (arch == Arch::vulkan) {
#ifdef TI_WITH_VULKAN
    vulkan::AotModuleParams vulkan_params =
        std::any_cast<vulkan::AotModuleParams &>(mod_params);
    return vulkan::make_aot_module(vulkan_params);
#endif
  } else if (arch == Arch::metal) {
#ifdef TI_WITH_METAL
    metal::AotModuleParams metal_params =
        std::any_cast<metal::AotModuleParams &>(mod_params);
    return metal::make_aot_module(metal_params);
#endif
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

}  // namespace aot
}  // namespace lang
}  // namespace taichi
