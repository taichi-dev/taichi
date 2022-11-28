#include "taichi/rhi/vulkan/vulkan_event.h"

namespace taichi::lang {
namespace vulkan {

VulkanDeviceEvent::VulkanDeviceEvent(vkapi::IVkEvent event) : vkapi_ref(event) {
}
VulkanDeviceEvent::~VulkanDeviceEvent() {
}

}  // namespace vulkan
}  // namespace taichi::lang
