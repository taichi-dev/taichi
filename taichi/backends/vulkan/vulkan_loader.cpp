#pragma once

#include "taichi/backends/vulkan/vulkan_common.h"

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
#if defined(__APPLE__) || defined(TI_EMSCRIPTENED)
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
#if defined(__APPLE__) || defined(TI_EMSCRIPTENED)
#else
  volkLoadInstance(instance);
#endif
}
void VulkanLoader::load_device(VkDevice device) {
  vulkan_device_ = device;
#if defined(__APPLE__) || defined(TI_EMSCRIPTENED)
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

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
