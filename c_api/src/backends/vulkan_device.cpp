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

// Taichi_DeviceAllocation *vulkan_allocate_memory(
//     Taichi_Device *dev,
//     const Taichi_DeviceAllocParams *params) {
//   tl::Device::AllocParams aparams;
//   aparams.size = params->size;
//   aparams.host_write = params->host_write;
//   aparams.host_read = params->host_read;
//   aparams.export_sharing = params->export_sharing;
//   aparams.usage = tl::AllocUsage::Storage;
//   auto *res = new tl::DeviceAllocation();
//   *res = cppcast(dev)->allocate_memory(aparams);
//   return reinterpret_cast<Taichi_DeviceAllocation *>(res);
// }

// void vulkan_dealloc_memory(Taichi_Device *dev, Taichi_DeviceAllocation *da) {
//   auto *alloc = cppcast(da);
//   cppcast(dev)->dealloc_memory(*alloc);
//   delete alloc;
// }

// void *vulkan_map_memory(Taichi_Device *dev, Taichi_DeviceAllocation *da) {
//   tl::DeviceAllocation *alloc = cppcast(da);
//   return cppcast(dev)->map(*alloc);
// }

// void vulkan_unmap_memory(Taichi_Device *dev, Taichi_DeviceAllocation *da) {
//   tl::DeviceAllocation *alloc = cppcast(da);
//   cppcast(dev)->unmap(*alloc);
// }

// Taichi_AotModule *make_vulkan_aot_module(const char *module_path,
//                                          Taichi_VulkanRuntime *runtime) {
//   tl::vulkan::AotModuleParams params;
//   params.module_path = module_path;
//   params.runtime = cppcast(runtime);
//   auto mod = tvk::make_aot_module(params);
//   return reinterpret_cast<Taichi_AotModule *>(mod.release());
// }

// void destroy_vulkan_aot_module(Taichi_AotModule *m) {
//   delete cppcast(m);
// }
