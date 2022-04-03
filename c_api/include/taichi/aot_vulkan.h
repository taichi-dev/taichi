#pragma once

#include <stdint.h>

#include "c_api/include/taichi/aot_module.h"
#include "taichi/common/platform_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EmbeddedVulkanDevice;
typedef struct VulkanDevice;
typedef struct VulkanRuntime;

TI_DLL_EXPORT EmbeddedVulkanDevice *make_embedded_vulkan_device(
    uint32_t api_version,
    const char **instance_extensions,
    uint32_t instance_extensions_count,
    const char **device_extensions,
    uint32_t device_extensions_count);

TI_DLL_EXPORT void destroy_embedded_vulkan_device(EmbeddedVulkanDevice *evd);

TI_DLL_EXPORT VulkanDevice *get_vulkan_device(EmbeddedVulkanDevice *evd);

TI_DLL_EXPORT VulkanRuntime *make_vulkan_runtime(uint64_t *host_result_buffer,
                                                 VulkanDevice *vk_device);
TI_DLL_EXPORT void destroy_vulkan_runtime(VulkanRuntime *vr);
TI_DLL_EXPORT void vulkan_synchronize(VulkanRuntime *vr);

TI_DLL_EXPORT AotModule *make_vulkan_aot_module(const char *module_path,
                                                VulkanRuntime runtime);
TI_DLL_EXPORT void destroy_vulkan_aot_module(AotModule *m);

#ifdef __cplusplus
}  // extern "C"
#endif
