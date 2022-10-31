---
sidebar_positions: 2
---

# Vulkan Backend Features

Taichi's Vulkan API gives you further control over the Vulkan version and extension requirements and allows you to interop with external Vulkan applications with shared resources.

## API Reference

`structure.vulkan_runtime_interop_info`

Necessary detail to share the same Vulkan runtime between Taichi and external procedures.

- `structure.vulkan_runtime_interop_info.get_instance_proc_addr`: Pointer to Vulkan loader function `vkGetInstanceProcAddr`.
- `structure.vulkan_runtime_interop_info.api_version`: Target Vulkan API version.
- `structure.vulkan_runtime_interop_info.instance`: Vulkan instance handle.
- `structure.vulkan_runtime_interop_info.physical_device`: Vulkan physical device handle.
- `structure.vulkan_runtime_interop_info.device`: Vulkan logical device handle.
- `structure.vulkan_runtime_interop_info.compute_queue`: Vulkan queue handle created in the queue family at `structure.vulkan_runtime_interop_info.compute_queue_family_index`.
- `structure.vulkan_runtime_interop_info.compute_queue_family_index`: Index of a Vulkan queue family with the `VK_QUEUE_COMPUTE_BIT` set.
- `structure.vulkan_runtime_interop_info.graphics_queue`: Vulkan queue handle created in the queue family at `structure.vulkan_runtime_interop_info.graphics_queue_family_index`.
- `structure.vulkan_runtime_interop_info.graphics_queue_family_index`: Index of a Vulkan queue family with the `VK_QUEUE_GRAPHICS_BIT` set.

**NOTE** `structure.vulkan_runtime_interop_info.compute_queue` and `structure.vulkan_runtime_interop_info.graphics_queue` can be the same if the queue family have `VK_QUEUE_COMPUTE_BIT` and `VK_QUEUE_GRAPHICS_BIT` set at the same tiem.

`structure.vulkan_memory_interop_info`

Necessary detail to share the same piece of Vulkan buffer between Taichi and external procedures.

- `structure.vulkan_memory_interop_info.buffer`: Vulkan buffer.
- `structure.vulkan_memory_interop_info.size`: Size of the piece of memory in bytes.
- `structure.vulkan_memory_interop_info.usage`: Vulkan buffer usage. In most of the cases, Taichi requires the `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`.
- `structure.vulkan_memory_interop_info.memory`: Device memory binded to the Vulkan buffer.
- `structure.vulkan_memory_interop_info.offset`: Offset in `VkDeviceMemory` object to the beginning of this allocation, in bytes.

`structure.vulkan_image_interop_info`

Necessary detail to share the same piece of Vulkan image between Taichi and external procedures.

- `structure.vulkan_image_interop_info.image`: Vulkan image.
- `structure.vulkan_image_interop_info.image_type`: Vulkan image allocation type.
- `structure.vulkan_image_interop_info.format`: Pixel format.
- `structure.vulkan_image_interop_info.extent`: Image extent.
- `structure.vulkan_image_interop_info.mip_level_count`: Number of mip-levels of the image.
- `structure.vulkan_image_interop_info.array_layer_count`: Number of array layers.
- `structure.vulkan_image_interop_info.sample_count`: Number of samples per pixel.
- `structure.vulkan_image_interop_info.tiling`: Image tiling.
- `structure.vulkan_image_interop_info.usage`: Vulkan image usage. In most cases, Taichi requires the `VK_IMAGE_USAGE_STORAGE_BIT` and the `VK_IMAGE_USAGE_SAMPLED_BIT`.

`structure.vulkan_event_interop_info`

Necessary detail to share the same Vulkan event synchronization primitive between Taichi and the user application.

- `structure.vulkan_event_interop_info.event`: Vulkan event handle.

`function.create_vulkan_runtime`

Creates a Vulkan Taichi runtime with user-controlled capability settings.

`function.import_vulkan_runtime`

Imports the Vulkan runtime owned by Taichi to external procedures.

`function.export_vulkan_runtime`

Exports a Vulkan runtime from external procedures to Taichi.

`function.import_vulkan_memory`

Imports the Vulkan buffer owned by Taichi to external procedures.

`function.export_vulkan_memory`

Exports a Vulkan buffer from external procedures to Taichi.

`function.import_vulkan_image`

Imports the Vulkan image owned by Taichi to external procedures.

`function.export_vulkan_image`

Exports a Vulkan image from external procedures to Taichi.

`function.import_vulkan_event`

Imports the Vulkan event owned by Taichi to external procedures.

`function.export_vulkan_event`

Exports a Vulkan event from external procedures to Taichi.
