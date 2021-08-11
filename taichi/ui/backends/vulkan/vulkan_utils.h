#pragma once
#include "taichi/ui/utils/utils.h"
#include "platform_specific/platform.h"
#include <unordered_map>
#include <optional>
#include "taichi/backends/vulkan/vulkan_api.h"
#include "taichi/backends/vulkan/loader.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

struct MappedMemory {
  void *data;
  VkDevice device;
  VkDeviceMemory mem;
  VkDeviceSize size;
  MappedMemory(VkDevice device_, VkDeviceMemory mem_, VkDeviceSize size_)
      : device(device_), mem(mem_), size(size_) {
    vkMapMemory(device, mem, 0, size, 0, &data);
  }
  ~MappedMemory() {
    vkUnmapMemory(device, mem);
  }
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

SwapChainSupportDetails query_swap_chain_support(
    VkPhysicalDevice physical_device,
    VkSurfaceKHR surface);

VkFormat find_supported_format(const std::vector<VkFormat> &candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features,
                               VkPhysicalDevice physical_device);

VkFormat find_depth_format(VkPhysicalDevice physical_device);

VkCommandBuffer create_new_command_buffer(VkCommandPool command_pool,
                                          VkDevice device);

VkCommandBuffer begin_single_time_commands(VkCommandPool command_pool,
                                           VkDevice device);

void end_single_time_commands(VkCommandBuffer command_buffer,
                              VkCommandPool command_pool,
                              VkDevice device,
                              VkQueue graphics_queue);

void copy_buffer(VkBuffer src_buffer,
                 VkBuffer dst_buffer,
                 VkDeviceSize size,
                 VkCommandPool command_pool,
                 VkDevice device,
                 VkQueue graphics_queue);

void copy_buffer_to_image(VkBuffer buffer,
                          VkImage image,
                          uint32_t width,
                          uint32_t height,
                          VkCommandPool command_pool,
                          VkDevice device,
                          VkQueue graphics_queue);

uint32_t find_memory_type(uint32_t type_filter,
                          VkMemoryPropertyFlags properties,
                          VkPhysicalDevice physical_device);

void alloc_device_memory(VkMemoryRequirements mem_requirements,
                         VkMemoryPropertyFlags properties,
                         VkDevice device,
                         VkPhysicalDevice physical_device,
                         VkDeviceMemory &mem);

void create_semaphore(VkSemaphore &result, VkDevice device);

void create_buffer(VkDeviceSize size,
                   VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags properties,
                   VkBuffer &buffer,
                   VkDeviceMemory &buffer_mem,
                   VkDevice device,
                   VkPhysicalDevice physical_device);

VkShaderModule create_shader_module(const std::vector<char> &code,
                                    VkDevice device);

void create_image(int dim,
                  uint32_t width,
                  uint32_t height,
                  uint32_t depth,
                  VkFormat format,
                  VkImageTiling tiling,
                  VkImageUsageFlags usage,
                  VkMemoryPropertyFlags properties,
                  VkImage &image,
                  VkDeviceMemory &image_mem,
                  VkDevice device,
                  VkPhysicalDevice physical_device);

VkImageView create_image_view(int dim,
                              VkImage image,
                              VkFormat format,
                              VkImageAspectFlags aspect_flags,
                              VkDevice device);

void transition_image_layout(VkImage image,
                             VkFormat format,
                             VkImageLayout old_layout,
                             VkImageLayout new_layout,
                             VkCommandPool command_pool,
                             VkDevice device,
                             VkQueue graphics_queue);

}  // namespace vulkan

TI_UI_NAMESPACE_END
