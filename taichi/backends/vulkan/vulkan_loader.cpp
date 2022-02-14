#include "taichi/backends/vulkan/vulkan_common.h"

#include "taichi/lang_util.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "taichi/common/logging.h"

namespace taichi {
namespace lang {
namespace vulkan {

VulkanLoader::VulkanLoader() {
}

bool VulkanLoader::init() {
  std::call_once(init_flag_, [&]() {
    if (initialized) {
      return;
    }
#if defined(TI_EMSCRIPTENED)
    initialized = true;
#elif defined(__APPLE__)
    vulkan_rt_ = std::make_unique<DynamicLoader>(runtime_lib_dir() + "/libMoltenVK.dylib");
    PFN_vkGetInstanceProcAddr get_proc_addr = (PFN_vkGetInstanceProcAddr)vulkan_rt_->load_function("vkGetInstanceProcAddr");

    volkInitializeCustom(get_proc_addr);
    initialized = true;
#else
    VkResult result = volkInitialize();
    initialized = result == VK_SUCCESS;
#endif
  });
  return initialized;
}

void VulkanLoader::load_instance(VkInstance instance) {
  vulkan_instance_ = instance;
#if defined(TI_EMSCRIPTENED)
#else
  volkLoadInstance(instance);
#endif
}
void VulkanLoader::load_device(VkDevice device) {
  vulkan_device_ = device;
#if defined(TI_EMSCRIPTENED)
#else
  volkLoadDevice(device);
#endif
}

PFN_vkVoidFunction VulkanLoader::load_function(const char *name) {
  auto result =
      vkGetInstanceProcAddr(VulkanLoader::instance().vulkan_instance_, name);
  TI_WARN_IF(result == nullptr, "loaded vulkan function {} is nullptr", name);
  return result;
}

bool is_vulkan_api_available() {
  return VulkanLoader::instance().init();
}

void set_vulkan_visible_device(std::string id) {
  VulkanLoader::instance().visible_device_id = id;
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
