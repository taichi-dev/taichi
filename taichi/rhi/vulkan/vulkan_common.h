#pragma once

#ifdef _WIN64
#define VK_USE_PLATFORM_WIN32_KHR 1
#endif

#ifdef ANDROID
#define VK_USE_PLATFORM_ANDROID_KHR
#endif

#include <volk.h>
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif  // VK_NO_PROTOTYPES

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include "taichi/rhi/device.h"

#include <stdexcept>

namespace taichi::lang {
namespace vulkan {

#define BAIL_ON_VK_BAD_RESULT(result, msg)               \
  do {                                                   \
    if ((result) != VK_SUCCESS) {                        \
      TI_ERROR("Vulkan Error : {} : {}", result, (msg)); \
    };                                                   \
  } while (0)

inline constexpr VkAllocationCallbacks *kNoVkAllocCallbacks = nullptr;

class VulkanDevice;
class VulkanSurface;
class VulkanStream;
class VulkanCommandList;
class VulkanResourceBinder;
class VulkanStreamSemaphoreObject;

struct VulkanRenderPassDesc;
struct VulkanFramebufferDesc;

}  // namespace vulkan
}  // namespace taichi::lang
