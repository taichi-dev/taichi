#pragma once

#include "taichi/taichi_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef TiOpaqueHandle TiDevice;

TI_DLL_EXPORT TiDevice tiLoadVulkanAotModule(const char* path);

TI_DLL_EXPORT TiDevice tiCreateVulkanDevice(TiContext context);

TI_DLL_EXPORT void tiImportVulkanDeviceAllocation(TiDevice device);

#ifdef __cplusplus
}  // extern "C"
#endif
