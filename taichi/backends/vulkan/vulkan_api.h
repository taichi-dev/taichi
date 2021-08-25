#pragma once

#ifdef _WIN64
#define VK_USE_PLATFORM_WIN32_KHR 1
#endif

#include <volk.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <taichi/backends/device.h>

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
class EmbeddedVulkanDevice {
 public:
  struct Params {
    std::optional<uint32_t> api_version;
    bool is_for_ui{false};
    std::vector<std::string> additional_instance_extensions;
    std::vector<std::string> additional_device_extensions;
    // the VkSurfaceKHR needs to be created after creating the VkInstance, but
    // before creating the VkPhysicalDevice thus, we allow the user to pass in a
    // custom surface creator
    std::function<VkSurfaceKHR(VkInstance)> surface_creator;
  };

  explicit EmbeddedVulkanDevice(const Params &params);
  ~EmbeddedVulkanDevice();

  VkInstance instance() {
    return instance_;
  }

  VulkanDevice *device() {
    return ti_device_.get();
  }

  const VulkanDevice *device() const {
    return ti_device_.get();
  }

  VkPhysicalDevice physical_device() const {
    return physical_device_;
  }

  VkSurfaceKHR surface() const {
    return surface_;
  }

  VkInstance instance() const {
    return instance_;
  }

  const VulkanQueueFamilyIndices &queue_family_indices() const {
    return queue_family_indices_;
  }

  Device *get_ti_device() const;

 private:
  void create_instance();
  void setup_debug_messenger();
  void create_surface();
  void pick_physical_device();
  void create_logical_device();

  VkInstance instance_{VK_NULL_HANDLE};
  VkDebugUtilsMessengerEXT debug_messenger_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VulkanQueueFamilyIndices queue_family_indices_;
  VkDevice device_{VK_NULL_HANDLE};
  // TODO: It's probably not right to put these per-queue things here. However,
  // in Taichi we only use a single queue on a single device (i.e. a single CUDA
  // stream), so it doesn't make a difference.
  VkQueue compute_queue_{VK_NULL_HANDLE};
  VkQueue graphics_queue_{VK_NULL_HANDLE};
  VkQueue present_queue_{VK_NULL_HANDLE};

  VkSurfaceKHR surface_{VK_NULL_HANDLE};

  // TODO: Shall we have dedicated command pools for COMPUTE and TRANSFER
  // commands, respectively?
  VkCommandPool command_pool_{VK_NULL_HANDLE};

  std::unique_ptr<VulkanDevice> ti_device_{nullptr};

  Params params_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
