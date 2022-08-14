---
sidebar_positions: 2
---

# Vulkan Backend Features

Taichi's Vulkan API gives you further control over Vulkan version and extension requirements and allows you to interop with external Vulkan applications with shared resources.

## API Reference

`structure.vulkan_runtime_interop_info`

Necessary detail to share a same Vulkan runtime between Taichi and user applications.

- `structure.vulkan_runtime_interop_info.api_version`: Targeted Vulkan API version.
- `structure.vulkan_runtime_interop_info.instance`: Vulkan instance handle.
- `structure.vulkan_runtime_interop_info.physical_device`: Vulkan physical device handle.
- `structure.vulkan_runtime_interop_info.device`: Vulkan logical device handle.
- `structure.vulkan_runtime_interop_info.compute_queue`: Vulkan queue handle created in the queue family at `structure.vulkan_runtime_interop_info.compute_queue_family_index`.
- `structure.vulkan_runtime_interop_info.compute_queue_family_index`: Index of a Vulkan queue family with the `VK_QUEUE_COMPUTE_BIT` set.
- `structure.vulkan_runtime_interop_info.graphics_queue`: Vulkan queue handle created in the queue family at `structure.vulkan_runtime_interop_info.graphics_queue_family_index`.
- `structure.vulkan_runtime_interop_info.graphics_queue_family_index`: Index of a Vulkan queue family with the `VK_QUEUE_GRAPHICS_BIT` set.

**NOTE** `structure.vulkan_runtime_interop_info.compute_queue` and `structure.vulkan_runtime_interop_info.graphics_queue` can be the same if the queue family have `VK_QUEUE_COMPUTE_BIT` and `VK_QUEUE_GRAPHICS_BIT` set at the same tiem.

`structure.vulkan_memory_interop_info`

Necessary detail to share a same piece of Vulkan buffer between Taichi and user applications.

- `structure.vulkan_memory_interop_info.buffer`: Vulkan buffer.
- `structure.vulkan_memory_interop_info.size`: Size of the piece of memory in bytes.
- `structure.vulkan_memory_interop_info.size`: Vulkan buffer usage. You usually want the `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` set.

`structure.vulkan_event_interop_info`

Necessary detail to share a same Vulkan event synchronization primitive between Taichi and user application.

- `structure.vulkan_event_interop_info.event`: Vulkan event handle.

`function.create_vulkan_runtime`

Create a Vulkan Taichi runtime with user controlled capability settings.

`function.import_vulkan_runtime`

Import the Vulkan runtime owned by Taichi to external user applications.

`function.export_vulkan_runtime`

Export a Vulkan runtime from external user applications to Taichi.

`function.import_vulkan_memory`

Import the Vulkan buffer owned by Taichi to external user applications.

`function.export_vulkan_memory`

Export a Vulkan buffer from external user applications to Taichi.

`function.import_vulkan_event`

Import the Vulkan event owned by Taichi to external user applications.

`function.export_vulkan_event`

Export a Vulkan event from external user applications to Taichi.
