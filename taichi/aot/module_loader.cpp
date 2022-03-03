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
    vulkan::AotModuleParams vulkan_params =
        std::any_cast<vulkan::AotModuleParams &>(vulkan_params);
    return vulkan::make_aot_module(vulkan_params);
  } else if (arch == Arch::metal) {

    metal::AotModuleParams metal_params =
        std::any_cast<metal::AotModuleParams &>(mod_params);
    return metal::make_aot_module(metal_params);
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

}  // namespace aot
}  // namespace lang
}  // namespace taichi
