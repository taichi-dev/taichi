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
TI_DLL_EXPORT TiDevice tiCreateVulkanDeviceEXT(
    uint32_t api_version,
    const char **instance_extensions,
    uint32_t instance_extensions_count,
    const char **device_extensions,
    uint32_t device_extensions_count);
TI_DLL_EXPORT TiDevice
tiImportVulkanDevice(const TiVulkanDeviceInteropInfo *importInfo);
TI_DLL_EXPORT void tiExportVulkanDevice(TiDevice device,
                                        TiVulkanDeviceInteropInfo *importInfo);

typedef struct TiVulkanDeviceAllocationInteropInfo {
  VkBuffer buffer;
} TiVulkanDeviceAllocationInteropInfo;
TI_DLL_EXPORT TiDeviceMemory tiImportVulkanDeviceMemory(
    TiDevice device,
    const TiVulkanDeviceAllocationInteropInfo *importInfo);
TI_DLL_EXPORT void tiExportVulkanDeviceMemory(
    TiDevice device,
    TiDeviceMemory deviceMemory,
    TiVulkanDeviceAllocationInteropInfo *importInfo);

TI_DLL_EXPORT TiAotModule tiLoadVulkanAotModule(TiContext context,
                                                const char *module_path);

#ifdef __cplusplus
}  // extern "C"
#endif
