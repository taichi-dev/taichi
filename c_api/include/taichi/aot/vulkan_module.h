#pragma once

#include <stdint.h>

#include "c_api/include/taichi/aot/module.h"
#include "c_api/include/taichi/backends/vulkan_device.h"
#include "taichi/common/platform_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

TI_DLL_EXPORT Taichi_AotModule *taichi_make_vulkan_aot_module(
    const char *module_path,
    Taichi_VulkanRuntime *runtime);

TI_DLL_EXPORT void taichi_destroy_vulkan_aot_module(Taichi_AotModule *m);

#ifdef __cplusplus
}  // extern "C"
#endif
