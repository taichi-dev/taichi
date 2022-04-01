#include "c_api/include/taichi/aot_vulkan.h"

#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/backends/vulkan/vulkan_device_creator.h"

namespace {
// Don't directly using namespace to avoid conflicting symbols.
namespace tvk = taichi::lang::vulkan;

tvk::VulkanDeviceCreator *cppcast(EmbeddedVulkanDevice *ptr) {
  return reinterpret_cast<tvk::VulkanDeviceCreator *>(ptr);
}

tvk::VkRuntime *cppcast(VulkanRuntime *ptr) {
  return reinterpret_cast<tvk::VkRuntime *>(ptr);
}

}  // namespace

EmbeddedVulkanDevice *make_embedded_vulkan_device(
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
  return reinterpret_cast<EmbeddedVulkanDevice *>(
      new tvk::VulkanDeviceCreator(params));
}

void destroy_embedded_vulkan_device(EmbeddedVulkanDevice *evd) {
  delete cppcast(evd);
}

VulkanDevice *get_vulkan_device(EmbeddedVulkanDevice *evd) {
  auto *ptr = cppcast(evd);
  return reinterpret_cast<VulkanDevice *>(ptr->device());
}

VulkanRuntime *make_vulkan_runtime(uint64_t *host_result_buffer,
                                   VulkanDevice *vk_device) {
  tvk::VkRuntime::Params params;
  params.host_result_buffer = host_result_buffer;
  params.device = reinterpret_cast<tvk::VulkanDevice *>(vk_device);
  return reinterpret_cast<VulkanRuntime *>(new tvk::VkRuntime(params));
}

void destroy_vulkan_runtime(VulkanRuntime *vr) {
  delete cppcast(vr);
}

void vulkan_synchronize(VulkanRuntime *vr) {
  cppcast(vr)->synchronize();
}
