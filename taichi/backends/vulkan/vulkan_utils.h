#pragma once

#include <volk.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace taichi {
namespace lang {
namespace vulkan {

class VulkanEnvSettings {
 public:
  static constexpr uint32_t kApiVersion() {
    return VK_API_VERSION_1_2;
  }
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
