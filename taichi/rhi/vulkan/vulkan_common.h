#pragma once

#include "taichi/rhi/impl_support.h"

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

#include <cstdio>

namespace taichi::lang {
namespace vulkan {

#define BAIL_ON_VK_BAD_RESULT_NO_RETURN(result, msg)                         \
  {                                                                          \
    if ((result) != VK_SUCCESS) {                                            \
      char vk_msg_buf[512];                                                  \
      std::snprintf(vk_msg_buf, sizeof(vk_msg_buf), "(%d) %s", result, msg); \
      RHI_LOG_ERROR(vk_msg_buf);                                             \
      RHI_ASSERT(false && "Error without return code");                      \
    };                                                                       \
  }

#define BAIL_ON_VK_BAD_RESULT(result, msg, retcode, retval)                  \
  {                                                                          \
    if ((result) != VK_SUCCESS) {                                            \
      char vk_msg_buf[512];                                                  \
      std::snprintf(vk_msg_buf, sizeof(vk_msg_buf), "(%d) %s", result, msg); \
      RHI_LOG_ERROR(vk_msg_buf);                                             \
      return {retcode, retval};                                              \
    };                                                                       \
  }

inline constexpr VkAllocationCallbacks *kNoVkAllocCallbacks = nullptr;

}  // namespace vulkan
}  // namespace taichi::lang
