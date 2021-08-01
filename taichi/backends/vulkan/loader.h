#pragma once
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

 private:
  bool initialized;

  VulkanLoader();

  VkInstance vulkan_instance;
  VkDevice vulkan_device;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
