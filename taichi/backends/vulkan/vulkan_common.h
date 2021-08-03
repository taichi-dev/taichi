#pragma once

#include <volk.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <stdexcept>

namespace taichi {
namespace lang {
namespace vulkan {

#pragma message("BAIL_ON_VK_BAD_RESULT uses exception")

#define BAIL_ON_VK_BAD_RESULT(result, msg) \
  do {                                     \
    if ((result) != VK_SUCCESS) {          \
      throw std::runtime_error((msg));     \
    };                                     \
  } while (0)

inline constexpr VkAllocationCallbacks *kNoVkAllocCallbacks = nullptr;

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
