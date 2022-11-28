#include "taichi/rhi/vulkan/vulkan_semaphore.h"

namespace taichi::lang {
namespace vulkan {

VulkanStreamSemaphoreObject::VulkanStreamSemaphoreObject(
    vkapi::IVkSemaphore sema)
    : vkapi_ref(sema) {
}
VulkanStreamSemaphoreObject::~VulkanStreamSemaphoreObject() {
}

}  // namespace vulkan
}  // namespace taichi::lang
