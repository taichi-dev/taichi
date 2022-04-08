#include "c_api/include/taichi/backends/vulkan_device.h"

#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/backends/vulkan/vulkan_device_creator.h"

namespace {

#include "c_api/src/inc/runtime_casts.inc.h"
#include "c_api/src/inc/vulkan_casts.inc.h"

}  // namespace

Taichi_EmbeddedVulkanDevice *taichi_make_embedded_vulkan_device(
    uint32_t api_version,
    const char **instance_extensions,
    uint32_t instance_extensions_count,
    const char **device_extensions,
    uint32_t device_extensions_count) {
  tvk::VulkanDeviceCreator::Params params;
  params.api_version = api_version;
  params.is_for_ui = false;
  params.additional_instance_extensions.reserve(instance_extensions_count);
  for (uint32_t i = 0; i < instance_extensions_count; ++i) {
    params.additional_instance_extensions.push_back(instance_extensions[i]);
  }
  params.additional_device_extensions.reserve(device_extensions_count);
  for (uint32_t i = 0; i < device_extensions_count; ++i) {
    params.additional_device_extensions.push_back(device_extensions[i]);
  }
  params.surface_creator = nullptr;
  return reinterpret_cast<Taichi_EmbeddedVulkanDevice *>(
      new tvk::VulkanDeviceCreator(params));
}

void taichi_destroy_embedded_vulkan_device(Taichi_EmbeddedVulkanDevice *evd) {
  delete cppcast(evd);
}

Taichi_VulkanDevice *taichi_get_vulkan_device(
    Taichi_EmbeddedVulkanDevice *evd) {
  auto *ptr = cppcast(evd);
  return reinterpret_cast<Taichi_VulkanDevice *>(ptr->device());
}

Taichi_VulkanRuntime *taichi_make_vulkan_runtime(
    uint64_t *host_result_buffer,
    Taichi_VulkanDevice *vk_device) {
  tvk::VkRuntime::Params params;
  params.host_result_buffer = host_result_buffer;
  params.device = cppcast(vk_device);
  return reinterpret_cast<Taichi_VulkanRuntime *>(new tvk::VkRuntime(params));
}

void taichi_destroy_vulkan_runtime(Taichi_VulkanRuntime *vr) {
  delete cppcast(vr);
}

void taichi_vulkan_add_root_buffer(Taichi_VulkanRuntime *vr,
                                   size_t root_buffer_size) {
  cppcast(vr)->add_root_buffer(root_buffer_size);
}

void taichi_vulkan_synchronize(Taichi_VulkanRuntime *vr) {
  cppcast(vr)->synchronize();
}
