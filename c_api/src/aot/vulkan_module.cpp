#include "c_api/include/taichi/aot/vulkan_module.h"

#include "taichi/backends/vulkan/aot_module_loader_impl.h"
#include "taichi/backends/vulkan/runtime.h"

namespace {

#include "c_api/src/inc/aot_casts.inc.h"

namespace tvk = ::taichi::lang::vulkan;

tvk::VkRuntime *cppcast(Taichi_VulkanRuntime *ptr) {
  return reinterpret_cast<tvk::VkRuntime *>(ptr);
}

}  // namespace

Taichi_AotModule *taichi_make_vulkan_aot_module(const char *module_path,
                                                Taichi_VulkanRuntime *runtime) {
  tl::vulkan::AotModuleParams params;
  params.module_path = module_path;
  params.runtime = cppcast(runtime);
  auto mod = tvk::make_aot_module(params);
  return reinterpret_cast<Taichi_AotModule *>(mod.release());
}

void taichi_destroy_vulkan_aot_module(Taichi_AotModule *m) {
  delete cppcast(m);
}
