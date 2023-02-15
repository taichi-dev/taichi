// # Vulkan Backend Features
//
// Taichi's Vulkan API gives you further control over the Vulkan version and
// extension requirements and allows you to interop with external Vulkan
// applications with shared resources.
//
#pragma once

#ifndef TAICHI_H
#include "taichi.h"
#endif  // TAICHI_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Structure `TiVulkanRuntimeInteropInfo` (1.4.0)
//
// Necessary detail to share the same Vulkan runtime between Taichi and external
// procedures.
//
//
// **NOTE** `compute_queue` and `graphics_queue` can be the same if the queue
// family have `VK_QUEUE_COMPUTE_BIT` and `VK_QUEUE_GRAPHICS_BIT` set at the
// same tiem.
typedef struct TiVulkanRuntimeInteropInfo {
  // Pointer to Vulkan loader function `vkGetInstanceProcAddr`.
  PFN_vkGetInstanceProcAddr get_instance_proc_addr;
  // Target Vulkan API version.
  uint32_t api_version;
  // Vulkan instance handle.
  VkInstance instance;
  // Vulkan physical device handle.
  VkPhysicalDevice physical_device;
  // Vulkan logical device handle.
  VkDevice device;
  // Vulkan queue handle created in the queue family at
  // `structure.vulkan_runtime_interop_info.compute_queue_family_index`.
  VkQueue compute_queue;
  // Index of a Vulkan queue family with the `VK_QUEUE_COMPUTE_BIT` set.
  uint32_t compute_queue_family_index;
  // Vulkan queue handle created in the queue family at
  // `structure.vulkan_runtime_interop_info.graphics_queue_family_index`.
  VkQueue graphics_queue;
  // Index of a Vulkan queue family with the `VK_QUEUE_GRAPHICS_BIT` set.
  uint32_t graphics_queue_family_index;
} TiVulkanRuntimeInteropInfo;

// Structure `TiVulkanMemoryInteropInfo` (1.4.0)
//
// Necessary detail to share the same piece of Vulkan buffer between Taichi and
// external procedures.
typedef struct TiVulkanMemoryInteropInfo {
  // Vulkan buffer.
  VkBuffer buffer;
  // Size of the piece of memory in bytes.
  uint64_t size;
  // Vulkan buffer usage. In most of the cases, Taichi requires the
  // `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`.
  VkBufferUsageFlags usage;
  // Device memory binded to the Vulkan buffer.
  VkDeviceMemory memory;
  // Offset in `VkDeviceMemory` object to the beginning of this allocation, in
  // bytes.
  uint64_t offset;
} TiVulkanMemoryInteropInfo;

// Structure `TiVulkanImageInteropInfo` (1.4.0)
//
// Necessary detail to share the same piece of Vulkan image between Taichi and
// external procedures.
typedef struct TiVulkanImageInteropInfo {
  // Vulkan image.
  VkImage image;
  // Vulkan image allocation type.
  VkImageType image_type;
  // Pixel format.
  VkFormat format;
  // Image extent.
  VkExtent3D extent;
  // Number of mip-levels of the image.
  uint32_t mip_level_count;
  // Number of array layers.
  uint32_t array_layer_count;
  // Number of samples per pixel.
  VkSampleCountFlagBits sample_count;
  // Image tiling.
  VkImageTiling tiling;
  // Vulkan image usage. In most cases, Taichi requires the
  // `VK_IMAGE_USAGE_STORAGE_BIT` and the `VK_IMAGE_USAGE_SAMPLED_BIT`.
  VkImageUsageFlags usage;
} TiVulkanImageInteropInfo;

// Function `ti_create_vulkan_runtime_ext` (1.4.0)
//
// Creates a Vulkan Taichi runtime with user-controlled capability settings.
TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_create_vulkan_runtime_ext(uint32_t api_version,
                             uint32_t instance_extension_count,
                             const char **instance_extensions,
                             uint32_t device_extension_count,
                             const char **device_extensions);

// Function `ti_import_vulkan_runtime` (1.4.0)
//
// Imports the Vulkan runtime owned by Taichi to external procedures.
TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_import_vulkan_runtime(const TiVulkanRuntimeInteropInfo *interop_info);

// Function `ti_export_vulkan_runtime` (1.4.0)
//
// Exports a Vulkan runtime from external procedures to Taichi.
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_runtime(TiRuntime runtime,
                         TiVulkanRuntimeInteropInfo *interop_info);

// Function `ti_import_vulkan_memory` (1.4.0)
//
// Imports the Vulkan buffer owned by Taichi to external procedures.
TI_DLL_EXPORT TiMemory TI_API_CALL
ti_import_vulkan_memory(TiRuntime runtime,
                        const TiVulkanMemoryInteropInfo *interop_info);

// Function `ti_export_vulkan_memory` (1.4.0)
//
// Exports a Vulkan buffer from external procedures to Taichi.
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_memory(TiRuntime runtime,
                        TiMemory memory,
                        TiVulkanMemoryInteropInfo *interop_info);

// Function `ti_import_vulkan_image` (1.4.0)
//
// Imports the Vulkan image owned by Taichi to external procedures.
TI_DLL_EXPORT TiImage TI_API_CALL
ti_import_vulkan_image(TiRuntime runtime,
                       const TiVulkanImageInteropInfo *interop_info,
                       VkImageViewType view_type,
                       VkImageLayout layout);

// Function `ti_export_vulkan_image` (1.4.0)
//
// Exports a Vulkan image from external procedures to Taichi.
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_image(TiRuntime runtime,
                       TiImage image,
                       TiVulkanImageInteropInfo *interop_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
