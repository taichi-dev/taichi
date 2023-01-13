---
sidebar_positions: 2
---

# Vulkan Backend Features

Taichi's Vulkan API gives you further control over the Vulkan version and extension requirements and allows you to interop with external Vulkan applications with shared resources.

## API Reference

### Structure `TiVulkanRuntimeInteropInfo`

> Stable since Taichi version: 1.4.0

```c
// structure.vulkan_runtime_interop_info
typedef struct TiVulkanRuntimeInteropInfo {
  PFN_vkGetInstanceProcAddr get_instance_proc_addr;
  uint32_t api_version;
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue compute_queue;
  uint32_t compute_queue_family_index;
  VkQueue graphics_queue;
  uint32_t graphics_queue_family_index;
} TiVulkanRuntimeInteropInfo;
```

Necessary detail to share the same Vulkan runtime between Taichi and external procedures.

- `get_instance_proc_addr`: Pointer to Vulkan loader function `vkGetInstanceProcAddr`.
- `api_version`: Target Vulkan API version.
- `instance`: Vulkan instance handle.
- `physical_device`: Vulkan physical device handle.
- `device`: Vulkan logical device handle.
- `compute_queue`: Vulkan queue handle created in the queue family at `compute_queue_family_index`.
- `compute_queue_family_index`: Index of a Vulkan queue family with the `VK_QUEUE_COMPUTE_BIT` set.
- `graphics_queue`: Vulkan queue handle created in the queue family at `graphics_queue_family_index`.
- `graphics_queue_family_index`: Index of a Vulkan queue family with the `VK_QUEUE_GRAPHICS_BIT` set.

**NOTE** `compute_queue` and `graphics_queue` can be the same if the queue family have `VK_QUEUE_COMPUTE_BIT` and `VK_QUEUE_GRAPHICS_BIT` set at the same tiem.

---
### Structure `TiVulkanMemoryInteropInfo`

> Stable since Taichi version: 1.4.0

```c
// structure.vulkan_memory_interop_info
typedef struct TiVulkanMemoryInteropInfo {
  VkBuffer buffer;
  uint64_t size;
  VkBufferUsageFlags usage;
  VkDeviceMemory memory;
  uint64_t offset;
} TiVulkanMemoryInteropInfo;
```

Necessary detail to share the same piece of Vulkan buffer between Taichi and external procedures.

- `buffer`: Vulkan buffer.
- `size`: Size of the piece of memory in bytes.
- `usage`: Vulkan buffer usage. In most of the cases, Taichi requires the `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`.
- `memory`: Device memory binded to the Vulkan buffer.
- `offset`: Offset in `VkDeviceMemory` object to the beginning of this allocation, in bytes.

---
### Structure `TiVulkanImageInteropInfo`

> Stable since Taichi version: 1.4.0

```c
// structure.vulkan_image_interop_info
typedef struct TiVulkanImageInteropInfo {
  VkImage image;
  VkImageType image_type;
  VkFormat format;
  VkExtent3D extent;
  uint32_t mip_level_count;
  uint32_t array_layer_count;
  VkSampleCountFlagBits sample_count;
  VkImageTiling tiling;
  VkImageUsageFlags usage;
} TiVulkanImageInteropInfo;
```

Necessary detail to share the same piece of Vulkan image between Taichi and external procedures.

- `image`: Vulkan image.
- `image_type`: Vulkan image allocation type.
- `format`: Pixel format.
- `extent`: Image extent.
- `mip_level_count`: Number of mip-levels of the image.
- `array_layer_count`: Number of array layers.
- `sample_count`: Number of samples per pixel.
- `tiling`: Image tiling.
- `usage`: Vulkan image usage. In most cases, Taichi requires the `VK_IMAGE_USAGE_STORAGE_BIT` and the `VK_IMAGE_USAGE_SAMPLED_BIT`.

---
### Function `ti_create_vulkan_runtime_ext`

> Stable since Taichi version: 1.4.0

```c
// function.create_vulkan_runtime
TI_DLL_EXPORT TiRuntime TI_API_CALL ti_create_vulkan_runtime_ext(
  uint32_t api_version,
  uint32_t instance_extension_count,
  const char** instance_extensions,
  uint32_t device_extension_count,
  const char** device_extensions
);
```

Creates a Vulkan Taichi runtime with user-controlled capability settings.

---
### Function `ti_import_vulkan_runtime`

> Stable since Taichi version: 1.4.0

```c
// function.import_vulkan_runtime
TI_DLL_EXPORT TiRuntime TI_API_CALL ti_import_vulkan_runtime(
  const TiVulkanRuntimeInteropInfo* interop_info
);
```

Imports the Vulkan runtime owned by Taichi to external procedures.

---
### Function `ti_export_vulkan_runtime`

> Stable since Taichi version: 1.4.0

```c
// function.export_vulkan_runtime
TI_DLL_EXPORT void TI_API_CALL ti_export_vulkan_runtime(
  TiRuntime runtime,
  TiVulkanRuntimeInteropInfo* interop_info
);
```

Exports a Vulkan runtime from external procedures to Taichi.

---
### Function `ti_import_vulkan_memory`

> Stable since Taichi version: 1.4.0

```c
// function.import_vulkan_memory
TI_DLL_EXPORT TiMemory TI_API_CALL ti_import_vulkan_memory(
  TiRuntime runtime,
  const TiVulkanMemoryInteropInfo* interop_info
);
```

Imports the Vulkan buffer owned by Taichi to external procedures.

---
### Function `ti_export_vulkan_memory`

> Stable since Taichi version: 1.4.0

```c
// function.export_vulkan_memory
TI_DLL_EXPORT void TI_API_CALL ti_export_vulkan_memory(
  TiRuntime runtime,
  TiMemory memory,
  TiVulkanMemoryInteropInfo* interop_info
);
```

Exports a Vulkan buffer from external procedures to Taichi.

---
### Function `ti_import_vulkan_image`

> Stable since Taichi version: 1.4.0

```c
// function.import_vulkan_image
TI_DLL_EXPORT TiImage TI_API_CALL ti_import_vulkan_image(
  TiRuntime runtime,
  const TiVulkanImageInteropInfo* interop_info,
  VkImageViewType view_type,
  VkImageLayout layout
);
```

Imports the Vulkan image owned by Taichi to external procedures.

---
### Function `ti_export_vulkan_image`

> Stable since Taichi version: 1.4.0

```c
// function.export_vulkan_image
TI_DLL_EXPORT void TI_API_CALL ti_export_vulkan_image(
  TiRuntime runtime,
  TiImage image,
  TiVulkanImageInteropInfo* interop_info
);
```

Exports a Vulkan image from external procedures to Taichi.
