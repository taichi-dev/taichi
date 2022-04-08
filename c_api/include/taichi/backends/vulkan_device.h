#pragma once

#include <stdint.h>

#include "c_api/include/taichi/backends/device_api.h"
#include "c_api/include/taichi/runtime.h"
#include "taichi/common/platform_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Taichi_EmbeddedVulkanDevice Taichi_EmbeddedVulkanDevice;
typedef struct Taichi_VulkanRuntime Taichi_VulkanRuntime;
typedef struct Taichi_VulkanDevice Taichi_VulkanDevice;

TI_DLL_EXPORT Taichi_EmbeddedVulkanDevice *taichi_make_embedded_vulkan_device(
    uint32_t api_version,
    const char **instance_extensions,
    uint32_t instance_extensions_count,
    const char **device_extensions,
    uint32_t device_extensions_count);

TI_DLL_EXPORT void taichi_destroy_embedded_vulkan_device(
    Taichi_EmbeddedVulkanDevice *evd);

TI_DLL_EXPORT Taichi_VulkanDevice *taichi_get_vulkan_device(
    Taichi_EmbeddedVulkanDevice *evd);

TI_DLL_EXPORT Taichi_VulkanRuntime *taichi_make_vulkan_runtime(
    uint64_t *host_result_buffer,
    Taichi_VulkanDevice *vk_device);

TI_DLL_EXPORT void taichi_destroy_vulkan_runtime(Taichi_VulkanRuntime *vr);

TI_DLL_EXPORT void taichi_vulkan_add_root_buffer(Taichi_VulkanRuntime *vr,
                                                 size_t root_buffer_size);

TI_DLL_EXPORT void taichi_vulkan_synchronize(Taichi_VulkanRuntime *vr);

#ifdef __cplusplus
}  // extern "C"
#endif
