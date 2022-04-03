#include "c_api/include/taichi/aot_vulkan.h"

#include "taichi/backends/vulkan/aot_module_loader_impl.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/backends/vulkan/vulkan_device_creator.h"

namespace {

#include "c_api/src/inc/runtime_casts.inc.h"
#include "c_api/src/inc/vulkan_casts.inc.h"

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
  params.device = cppcast(vk_device);
  return reinterpret_cast<VulkanRuntime *>(new tvk::VkRuntime(params));
}

void destroy_vulkan_runtime(VulkanRuntime *vr) {
  delete cppcast(vr);
}

void vulkan_add_root_buffer(VulkanRuntime *vr, size_t root_buffer_size) {
  cppcast(vr)->add_root_buffer(root_buffer_size);
}

void vulkan_synchronize(VulkanRuntime *vr) {
  cppcast(vr)->synchronize();
}

DeviceAllocation *vulkan_allocate_memory(VulkanDevice *dev,
                                         const DeviceAllocParams *params) {
  tl::Device::AllocParams aparams;
  aparams.size = params->size;
  aparams.host_write = params->host_write;
  aparams.host_read = params->host_read;
  aparams.export_sharing = params->export_sharing;
  aparams.usage = tl::AllocUsage::Storage;
  auto *res = new tl::DeviceAllocation();
  *res = cppcast(dev)->allocate_memory(aparams);
  return reinterpret_cast<DeviceAllocation *>(res);
}

void vulkan_dealloc_memory(VulkanDevice *dev, DeviceAllocation *da) {
  auto *alloc = cppcast(da);
  cppcast(dev)->dealloc_memory(*alloc);
  delete alloc;
}

void *vulkan_map_memory(VulkanDevice *dev, DeviceAllocation *da) {
  tl::DeviceAllocation *alloc = cppcast(da);
  return cppcast(dev)->map(*alloc);
}

void vulkan_unmap_memory(VulkanDevice *dev, DeviceAllocation *da) {
  tl::DeviceAllocation *alloc = cppcast(da);
  cppcast(dev)->unmap(*alloc);
}

AotModule *make_vulkan_aot_module(const char *module_path,
                                  VulkanRuntime *runtime) {
  tl::vulkan::AotModuleParams params;
  params.module_path = module_path;
  params.runtime = cppcast(runtime);
  auto mod = tvk::make_aot_module(params);
  return reinterpret_cast<AotModule *>(mod.release());
}

void destroy_vulkan_aot_module(AotModule *m) {
  delete cppcast(m);
}
