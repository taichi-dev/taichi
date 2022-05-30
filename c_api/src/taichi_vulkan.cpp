#include "taichi/taichi_core.h"
#include "taichi/taichi_vulkan.h"

#include "taichi/runtime/vulkan/runtime.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/backends/vulkan/vulkan_device_creator.h"

TiVulkanDevice taichi_make_embedded_vulkan_device(
    uint32_t api_version,
    const char **instance_extensions,
    uint32_t instance_extensions_count,
    const char **device_extensions,
    uint32_t device_extensions_count) {
  taichi::lang::vulkan::VulkanDeviceCreator::Params params;
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
  return reinterpret_cast<TiDevice>(
      new taichi::lang::vulkan::VulkanDeviceCreator(params));
}

void taichi_destroy_embedded_vulkan_device(TiDevice device) {
  delete (taichi::lang::vulkan::VulkanDevice*)(device);
}

TiDevice taichi_get_vulkan_device(TiDevice device) {
  auto *ptr = (taichi::lang::vulkan::VulkanDevice*)(device);
  return reinterpret_cast<TiDevice>(ptr->vk_device());
}

TiDevice taichi_make_vulkan_runtime(
    uint64_t *host_result_buffer,
    TiDevice vk_device) {
  taichi::lang::vulkan::VkRuntime::Params params;
  params.host_result_buffer = host_result_buffer;
  params.device = (taichi::lang::vulkan::VulkanDevice*)(vk_device);
  return new taichi::lang::vulkan::VkRuntime(params);
}

TiAotModule taichi_make_vulkan_aot_module(const char *module_path,
                                          TiVulkanRuntime runtime) {
  taichi::lang::vulkan::AotModuleParams params;
  params.module_path = module_path;
  params.runtime = (taichi::lang::vulkan::VkRuntime *)runtime;
  auto mod = taichi::lang::vulkan::make_aot_module(params);
  return mod.release();
}

void taichi_destroy_vulkan_aot_module(TiAotModule *aotModule) {
  delete (taichi::lang::aot::Module *)aotModule;
}
