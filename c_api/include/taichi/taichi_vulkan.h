#pragma once

#include "taichi/taichi_core.h"
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TiVulkanDeviceInteropInfo {
  VkInstance instance;
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkQueue computeQueue;
  uint32_t computeQueueFamilyIndex;
  VkQueue graphicsQueue;
  uint32_t graphicsQueueFamilyIndex;
} TiVulkanDeviceInteropInfo;
TI_DLL_EXPORT TiDevice
ti_create_vulkan_device_ext(uint32_t api_version,
                            const char **instance_extensions,
                            uint32_t instance_extensions_count,
                            const char **device_extensions,
                            uint32_t device_extensions_count);
TI_DLL_EXPORT TiDevice
ti_import_vulkan_device(const TiVulkanDeviceInteropInfo *interop_info);
TI_DLL_EXPORT void ti_export_vulkan_device(
    TiDevice device,
    TiVulkanDeviceInteropInfo *interop_info);

typedef struct TiVulkanDeviceMemoryInteropInfo {
  VkBuffer buffer;
  size_t size;
  VkBufferUsageFlags usage;
} TiVulkanDeviceMemoryInteropInfo;
TI_DLL_EXPORT TiDeviceMemory ti_import_vulkan_device_memory(
    TiDevice device,
    const TiVulkanDeviceMemoryInteropInfo *interop_info);
TI_DLL_EXPORT void ti_export_vulkan_device_memory(
    TiDevice device,
    TiDeviceMemory device_memory,
    TiVulkanDeviceMemoryInteropInfo *interop_info);

TI_DLL_EXPORT TiAotModule ti_load_vulkan_aot_module(TiContext context,
                                                    const char *module_path);

#ifdef __cplusplus
}  // extern "C"
#endif
