#pragma once
#include "taichi/ui/utils/utils.h"
#include "platform_specific/platform.h"
#include <unordered_map>
#include <optional>
#include "taichi/backends/vulkan/vulkan_api.h"
#include "taichi/backends/vulkan/loader.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

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

void create_semaphore(VkSemaphore &result, VkDevice device);

}  // namespace vulkan

TI_UI_NAMESPACE_END
