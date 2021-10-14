#pragma once

#include <thread>
#include <mutex>

#include <volk.h>

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

 private:
  std::once_flag init_flag_;
  bool initialized{false};

  VulkanLoader();

  VkInstance vulkan_instance_{VK_NULL_HANDLE};
  VkDevice vulkan_device_{VK_NULL_HANDLE};
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
