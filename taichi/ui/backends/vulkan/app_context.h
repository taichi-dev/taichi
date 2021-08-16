#pragma once
#include "taichi/ui/backends/vulkan/vulkan_utils.h"
#include "taichi/ui/common/app_config.h"
#include <memory>
#include "taichi/backends/vulkan/vulkan_api.h"
#include "taichi/backends/vulkan/loader.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class AppContext {
 public:
  void init(GLFWwindow *glfw_window);

  void cleanup_swap_chain();
  void cleanup();
  void recreate_swap_chain();

  int get_swap_chain_size();

  VkInstance instance() const;
  VkDevice device() const;
  VkPhysicalDevice physical_device() const;
  taichi::lang::vulkan::VulkanQueueFamilyIndices queue_family_indices() const;
  VkQueue graphics_queue() const;
  VkQueue present_queue() const;
  VkCommandPool command_pool() const;
  VkRenderPass render_pass() const;
  SwapChain &swap_chain();
  const SwapChain &swap_chain() const;
  GLFWwindow *glfw_window() const;

  AppConfig config;

 private:
  std::unique_ptr<taichi::lang::vulkan::EmbeddedVulkanDevice> vulkan_device_{
      nullptr};

  VkRenderPass render_pass_{VK_NULL_HANDLE};

  void create_render_passes();

  SwapChain swap_chain_;

  GLFWwindow *glfw_window_{nullptr};
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
