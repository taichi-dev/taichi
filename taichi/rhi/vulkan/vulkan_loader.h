#pragma once

#include <thread>
#include <mutex>

#include "taichi/rhi/vulkan/vulkan_common.h"
#include "taichi/system/dynamic_loader.h"

namespace taichi {
namespace lang {
namespace vulkan {

class TI_DLL_EXPORT VulkanLoader {
 public:
  static VulkanLoader &instance() {
    static VulkanLoader instance;
    return instance;
  }

 public:
  VulkanLoader(VulkanLoader const &) = delete;
  void operator=(VulkanLoader const &) = delete;

  bool check_vulkan_device();

  void load_instance(VkInstance instance_);
  void load_device(VkDevice device_);
  bool init();
  PFN_vkVoidFunction load_function(const char *name);
  VkInstance get_instance() {
    return vulkan_instance_;
  }
  std::string visible_device_id;

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

TI_DLL_EXPORT bool is_vulkan_api_available();

TI_DLL_EXPORT void set_vulkan_visible_device(std::string id);

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
