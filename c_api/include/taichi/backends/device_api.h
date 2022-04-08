#pragma once

#include <stdint.h>

#include "taichi/common/platform_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef TI_DLL_EXPORT struct {
  uint64_t size;
  bool host_write;
  bool host_read;
  bool export_sharing;
  // AllocUsage is an enum class, so not exported to C yet
} Taichi_DeviceAllocParams;

typedef struct Taichi_Device Taichi_Device;
typedef struct Taichi_DeviceAllocation Taichi_DeviceAllocation;

TI_DLL_EXPORT Taichi_DeviceAllocation *taichi_allocate_device_memory(
    Taichi_Device *dev,
    const Taichi_DeviceAllocParams *params);

TI_DLL_EXPORT void taichi_deallocate_device_memory(Taichi_Device *dev,
                                                   Taichi_DeviceAllocation *da);

TI_DLL_EXPORT void *taichi_map_device_allocation(Taichi_Device *dev,
                                                 Taichi_DeviceAllocation *da);

TI_DLL_EXPORT void taichi_unmap_device_allocation(Taichi_Device *dev,
                                                  Taichi_DeviceAllocation *da);

#ifdef __cplusplus
}
#endif
