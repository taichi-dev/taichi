#pragma once

#ifdef _WIN64
#define VK_USE_PLATFORM_WIN32_KHR 1
#endif

#include "taichi/rhi/vulkan/vulkan_common.h"

#include <taichi/rhi/device.h>

#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <functional>

namespace taichi {
namespace lang {
namespace vulkan {

class VulkanDevice;

struct VulkanQueueFamilyIndices {
  std::optional<uint32_t> compute_family;
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;
  // TODO: While it is the case that all COMPUTE/GRAPHICS queue also support
  // TRANSFER by default, maye there are some performance benefits to find a
  // TRANSFER-dedicated queue family.
  // https://vulkan-tutorial.com/Vertex_buffers/Staging_buffer#page_Transfer-queue

  bool is_complete() const {
    return compute_family.has_value();
  }

  bool is_complete_for_ui() {
    return graphics_family.has_value() && present_family.has_value();
  }
};

/**
 * This class creates a VulkanDevice instance. The underlying Vk* resources are
 * embedded directly inside the class.
 */
class TI_DLL_EXPORT VulkanDeviceCreator {
 public:
  struct Params {
    // User-provided API version. If assigned, the users MUST list all
    // their desired extensions in `additional_instance_extensions` and
    // `additional_device_extensions`; no extension is enabled by default.
    std::optional<uint32_t> api_version;
    bool is_for_ui{false};
    std::vector<std::string> additional_instance_extensions;
    std::vector<std::string> additional_device_extensions;
    // the VkSurfaceKHR needs to be created after creating the VkInstance, but
    // before creating the VkPhysicalDevice thus, we allow the user to pass in a
    // custom surface creator
    std::function<VkSurfaceKHR(VkInstance)> surface_creator;
    bool enable_validation_layer{false};
  };

  explicit VulkanDeviceCreator(const Params &params);
  ~VulkanDeviceCreator();

  const VulkanDevice *device() const {
    return ti_device_.get();
  }

  VulkanDevice *device() {
    return ti_device_.get();
  }

  VkSurfaceKHR get_surface() {
    return surface_;
  }

 private:
  void create_instance(bool manual_create);
  void setup_debug_messenger();
  void create_surface();
  void pick_physical_device();
  void create_logical_device(bool manual_create);

  uint32_t api_version_{VK_API_VERSION_1_0};
  VkInstance instance_{VK_NULL_HANDLE};
  VkDebugUtilsMessengerEXT debug_messenger_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VulkanQueueFamilyIndices queue_family_indices_;
  VkDevice device_{VK_NULL_HANDLE};

  VkQueue compute_queue_{VK_NULL_HANDLE};
  VkQueue graphics_queue_{VK_NULL_HANDLE};

  VkSurfaceKHR surface_{VK_NULL_HANDLE};

  std::unique_ptr<VulkanDevice> ti_device_{nullptr};

  Params params_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
