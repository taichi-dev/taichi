#pragma once
#include "taichi/ui/backends/vulkan/vulkan_utils.h"
#include "taichi/ui/common/app_config.h"
#include <memory>
#include "taichi/backends/vulkan/vulkan_api.h"
#include "taichi/backends/vulkan/loader.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class AppContext {
 public:
  void init(GLFWwindow *glfw_window, const AppConfig &config);
  void cleanup();

  VkInstance instance() const;
  VkDevice device() const;
  VkPhysicalDevice physical_device() const;
  VkQueue graphics_queue() const;
  VkQueue present_queue() const;
  VkCommandPool command_pool() const;
  VkSurfaceKHR surface() const;
  taichi::lang::vulkan::VulkanQueueFamilyIndices queue_family_indices() const;
  GLFWwindow *glfw_window() const;

  taichi::lang::vulkan::VulkanDevice& vulkan_device();
  const taichi::lang::vulkan::VulkanDevice& vulkan_device() const;

  AppConfig config;

 private:
  std::unique_ptr<taichi::lang::vulkan::EmbeddedVulkanDevice> vulkan_device_{
      nullptr};
  // not owned
  GLFWwindow *glfw_window_{nullptr};
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
