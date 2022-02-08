#pragma once

#include <thread>
#include <mutex>

#include "taichi/backends/vulkan/vulkan_common.h"
#include "taichi/system/dynamic_loader.h"

namespace taichi {
namespace lang {
namespace vulkan {

class VulkanLoader {
 public:
  static VulkanLoader &instance() {
    static VulkanLoader instance;
    return instance;
  }

 public:
  VulkanLoader(VulkanLoader const &) = delete;
  void operator=(VulkanLoader const &) = delete;

  void load_instance(VkInstance instance_);
  void load_device(VkDevice device_);
  bool init();
  PFN_vkVoidFunction load_function(const char *name);
  VkInstance get_instance() {
    return vulkan_instance_;
  }

 private:
  std::once_flag init_flag_;
  bool initialized{false};

  VulkanLoader();

#if defined(__APPLE__)
  std::unique_ptr<DynamicLoader> vulkan_rt_{nullptr};
#endif

  VkInstance vulkan_instance_{VK_NULL_HANDLE};
  VkDevice vulkan_device_{VK_NULL_HANDLE};
};

bool is_vulkan_api_available();

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
