#pragma once
#include <taichi/taichi_core.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// structure.vulkan_runtime_interop_info
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

// structure.vulkan_memory_interop_info
typedef struct TiVulkanMemoryInteropInfo {
  VkBuffer buffer;
  uint64_t size;
  VkBufferUsageFlags usage;
} TiVulkanMemoryInteropInfo;

// structure.vulkan_texture_interop_info
typedef struct TiVulkanTextureInteropInfo {
  VkImage image;
  VkImageType image_type;
  VkFormat format;
  VkExtent3D extent;
  uint32_t mip_level_count;
  uint32_t array_layer_count;
  VkSampleCountFlagBits sample_count;
  VkImageTiling tiling;
  VkImageUsageFlags usage;
} TiVulkanTextureInteropInfo;

// structure.vulkan_event_interop_info
typedef struct TiVulkanEventInteropInfo {
  VkEvent event;
} TiVulkanEventInteropInfo;

// function.create_vulkan_runtime
TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_create_vulkan_runtime_ext(uint32_t api_version,
                             uint32_t instance_extension_count,
                             const char **instance_extensions,
                             uint32_t device_extension_count,
                             const char **device_extensions);

// function.import_vulkan_runtime
TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_import_vulkan_runtime(const TiVulkanRuntimeInteropInfo *interop_info);

// function.export_vulkan_runtime
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_runtime(TiRuntime runtime,
                         TiVulkanRuntimeInteropInfo *interop_info);

// function.import_vulkan_memory
TI_DLL_EXPORT TiMemory TI_API_CALL
ti_import_vulkan_memory(TiRuntime runtime,
                        const TiVulkanMemoryInteropInfo *interop_info);

// function.export_vulkan_memory
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_memory(TiRuntime runtime,
                        TiMemory memory,
                        TiVulkanMemoryInteropInfo *interop_info);

// function.import_vulkan_texture
TI_DLL_EXPORT TiTexture TI_API_CALL
ti_import_vulkan_texture(TiRuntime runtime,
                         const TiVulkanTextureInteropInfo *interop_info,
                         VkImageViewType view_type,
                         VkImageLayout layout);

// function.export_vulkan_texture
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_texture(TiRuntime runtime,
                         TiTexture texture,
                         TiVulkanTextureInteropInfo *interop_info);

// function.import_vulkan_event
TI_DLL_EXPORT TiEvent TI_API_CALL
ti_import_vulkan_event(TiRuntime runtime,
                       const TiVulkanEventInteropInfo *interop_info);

// function.export_vulkan_event
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_event(TiRuntime runtime,
                       TiEvent event,
                       TiVulkanEventInteropInfo *interop_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
