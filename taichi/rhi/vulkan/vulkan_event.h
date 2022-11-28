#pragma once
#include "taichi/rhi/vulkan/vulkan_api.h"

namespace taichi::lang {
namespace vulkan {

class VulkanDeviceEvent : public DeviceEvent {
 public:
  explicit VulkanDeviceEvent(vkapi::IVkEvent event);
  ~VulkanDeviceEvent() override;

  vkapi::IVkEvent vkapi_ref{nullptr};
};

}  // namespace vulkan
}  // namespace taichi::lang
