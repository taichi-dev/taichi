#pragma once

#include "taichi/taichi_core.h"
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TiVulkanRuntimeInteropInfo {
  uint32_t api_version;
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue compute_queue;
  uint32_t compute_queue_family_index;
  VkQueue graphics_queue;
  uint32_t graphics_queue_family_index;
} TiVulkanRuntimeInteropInfo;
TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_create_vulkan_runtime_ext(uint32_t api_version,
                             const char **instance_extensions,
                             uint32_t instance_extensions_count,
                             const char **runtime_extensions,
                             uint32_t runtime_extensions_count);
TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_import_vulkan_runtime(const TiVulkanRuntimeInteropInfo *interop_info);
TI_DLL_EXPORT void ti_export_vulkan_runtime(
    TiRuntime runtime,
    TiVulkanRuntimeInteropInfo *interop_info);

typedef struct TiVulkanMemoryInteropInfo {
  VkBuffer buffer;
  size_t size;
  VkBufferUsageFlags usage;
} TiVulkanMemoryInteropInfo;
TI_DLL_EXPORT TiMemory TI_API_CALL
ti_import_vulkan_memory(TiRuntime runtime,
                        const TiVulkanMemoryInteropInfo *interop_info);
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_memory(TiRuntime runtime,
                        TiMemory memory,
                        TiVulkanMemoryInteropInfo *interop_info);

#ifdef __cplusplus
}  // extern "C"
#endif
