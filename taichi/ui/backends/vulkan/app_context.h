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
  void init();

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

  AppConfig config;
  SwapChain swap_chain;

  GLFWwindow *glfw_window;

 private:
  std::unique_ptr<taichi::lang::vulkan::EmbeddedVulkanDevice> vulkan_device_;

  VkRenderPass render_pass_;

  void create_render_pass(VkRenderPass &render_pass,
                          VkImageLayout final_color_layout);
  void create_render_passes();
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
