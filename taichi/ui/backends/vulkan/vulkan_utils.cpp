#include "taichi/ui/backends/vulkan/vulkan_utils.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

SwapChainSupportDetails query_swap_chain_support(
    VkPhysicalDevice physical_device,
    VkSurfaceKHR surface) {
  SwapChainSupportDetails details;

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface,
                                            &details.capabilities);

  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count,
                                       nullptr);

  if (format_count != 0) {
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface,
                                         &format_count, details.formats.data());
  }

  uint32_t present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                            &present_mode_count, nullptr);

  if (present_mode_count != 0) {
    details.present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                              &present_mode_count,
                                              details.present_modes.data());
  }

  return details;
}

VkFormat find_supported_format(const std::vector<VkFormat> &candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features,
                               VkPhysicalDevice physical_device) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  throw std::runtime_error("failed to find supported format!");
}

VkFormat find_depth_format(VkPhysicalDevice physical_device) {
  return find_supported_format(
      {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
       VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
      physical_device);
}

void create_semaphore(VkSemaphore &result, VkDevice device) {
  VkSemaphoreCreateInfo semaphore_info = {};
  memset(&semaphore_info, 0, sizeof(semaphore_info));
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkExportSemaphoreCreateInfoKHR export_semaphore_create_info = {};
  export_semaphore_create_info.sType =
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;

#ifdef _WIN64
  WindowsSecurityAttributes win_security_attribs;

  VkExportSemaphoreWin32HandleInfoKHR export_semaphore_win32_handle_info = {};
  export_semaphore_win32_handle_info.sType =
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
  export_semaphore_win32_handle_info.pNext = NULL;
  export_semaphore_win32_handle_info.pAttributes = &win_security_attribs;
  export_semaphore_win32_handle_info.dwAccess =
      DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
  export_semaphore_win32_handle_info.name = (LPCWSTR)NULL;

  export_semaphore_create_info.pNext =
      IsWindows8OrGreater() ? &export_semaphore_win32_handle_info : NULL;
  export_semaphore_create_info.handleTypes =
      IsWindows8OrGreater()
          ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
          : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  export_semaphore_create_info.pNext = NULL;
  export_semaphore_create_info.handleTypes =
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
  semaphore_info.pNext = &export_semaphore_create_info;

  if (vkCreateSemaphore(device, &semaphore_info, nullptr, &result) !=
      VK_SUCCESS) {
    throw std::runtime_error(
        "failed to create synchronization objects for a CUDA-Vulkan!");
  }
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
