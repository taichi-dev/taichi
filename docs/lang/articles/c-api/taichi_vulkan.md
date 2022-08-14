---
sidebar_positions: 2
---

# Vulkan Backend Features

Taichi's Vulkan API gives you further control over Vulkan version and extension requirements and allows you to interop with external Vulkan applications with shared resources.

## API Reference

### Structure `TiVulkanRuntimeInteropInfo`

```c
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
```

Necessary detail to share a same Vulkan runtime between Taichi and user applications.

- `api_version`: Targeted Vulkan API version.
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

```c
// structure.vulkan_memory_interop_info
typedef struct TiVulkanMemoryInteropInfo {
  VkBuffer buffer;
  uint64_t size;
  VkBufferUsageFlags usage;
} TiVulkanMemoryInteropInfo;
```

Necessary detail to share a same piece of Vulkan buffer between Taichi and user applications.

- `buffer`: Vulkan buffer.
- `size`: Size of the piece of memory in bytes.
- `size`: Vulkan buffer usage. You usually want the `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` set.

---
### Structure `TiVulkanEventInteropInfo`

```c
// structure.vulkan_event_interop_info
typedef struct TiVulkanEventInteropInfo {
  VkEvent event;
} TiVulkanEventInteropInfo;
```

Necessary detail to share a same Vulkan event synchronization primitive between Taichi and user application.

- `event`: Vulkan event handle.

---
### Function `ti_create_vulkan_runtime_ext`

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

Create a Vulkan Taichi runtime with user controlled capability settings.

---
### Function `ti_import_vulkan_runtime`

```c
// function.import_vulkan_runtime
TI_DLL_EXPORT TiRuntime TI_API_CALL ti_import_vulkan_runtime(
  const TiVulkanRuntimeInteropInfo* interop_info
);
```

Import the Vulkan runtime owned by Taichi to external user applications.

---
### Function `ti_export_vulkan_runtime`

```c
// function.export_vulkan_runtime
TI_DLL_EXPORT void TI_API_CALL ti_export_vulkan_runtime(
  TiRuntime runtime,
  TiVulkanRuntimeInteropInfo* interop_info
);
```

Export a Vulkan runtime from external user applications to Taichi.

---
### Function `ti_import_vulkan_memory`

```c
// function.import_vulkan_memory
TI_DLL_EXPORT TiMemory TI_API_CALL ti_import_vulkan_memory(
  TiRuntime runtime,
  const TiVulkanMemoryInteropInfo* interop_info
);
```

Import the Vulkan buffer owned by Taichi to external user applications.

---
### Function `ti_export_vulkan_memory`

```c
// function.export_vulkan_memory
TI_DLL_EXPORT void TI_API_CALL ti_export_vulkan_memory(
  TiRuntime runtime,
  TiMemory memory,
  TiVulkanMemoryInteropInfo* interop_info
);
```

Export a Vulkan buffer from external user applications to Taichi.

---
### Function `ti_import_vulkan_event`

```c
// function.import_vulkan_event
TI_DLL_EXPORT TiEvent TI_API_CALL ti_import_vulkan_event(
  TiRuntime runtime,
  const TiVulkanEventInteropInfo* interop_info
);
```

Import the Vulkan event owned by Taichi to external user applications.

---
### Function `ti_export_vulkan_event`

```c
// function.export_vulkan_event
TI_DLL_EXPORT void TI_API_CALL ti_export_vulkan_event(
  TiRuntime runtime,
  TiEvent event,
  TiVulkanEventInteropInfo* interop_info
);
```

Export a Vulkan event from external user applications to Taichi.
