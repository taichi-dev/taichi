#pragma once
#include "taichi/rhi/vulkan/vulkan_api.h"

namespace taichi::lang {
namespace vulkan {

class VulkanStreamSemaphoreObject : public StreamSemaphoreObject {
 public:
  explicit VulkanStreamSemaphoreObject(vkapi::IVkSemaphore sema);
  ~VulkanStreamSemaphoreObject() override;

  vkapi::IVkSemaphore vkapi_ref{nullptr};
};

}  // namespace vulkan
}  // namespace taichi::lang
