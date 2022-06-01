#pragma once

#include "taichi/taichi_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t TiBool;
#define TI_TRUE 1u
#define TI_FALSE 0u

typedef uint32_t TiFlags;

typedef void *TiDispatchableHandle;
typedef size_t TiNonDispatchableHandle;
#define TI_NULL_HANDLE 0

typedef TiDispatchableHandle TiDevice;
typedef TiDispatchableHandle TiContext;
typedef TiDispatchableHandle TiAotModule;

typedef TiNonDispatchableHandle TiDeviceMemory;
typedef TiNonDispatchableHandle TiKernel;

typedef enum TiArch {
  // TI_ARCH_ANY = 0,  // TODO: (penguinliong) Do we need this?
  TI_ARCH_VULKAN = 1,
} TiArch;
TI_DLL_EXPORT TiDevice ti_create_device(TiArch arch);
TI_DLL_EXPORT void ti_destroy_device(TiDevice device);
TI_DLL_EXPORT void ti_device_wait_idle(TiDevice device);

typedef enum TiAllocationUsageFlagBits {
  TI_MEMORY_USAGE_STORAGE_BIT = 1,
  TI_MEMORY_USAGE_UNIFORM_BIT = 2,
  TI_MEMORY_USAGE_VERTEX_BIT = 4,
  TI_MEMORY_USAGE_INDEX_BIT = 8,
} TiMemoryUsageFlagBits;
typedef TiFlags TiMemoryUsageFlags;
typedef struct {
  uint64_t size;
  TiBool hostWrite;
  TiBool hostRead;
  TiBool exportSharing;
  TiMemoryUsageFlags usage;
} TiMemoryAllocateInfo;
TI_DLL_EXPORT TiDeviceMemory
ti_allocate_memory(TiDevice device, const TiMemoryAllocateInfo *allocateInfo);
TI_DLL_EXPORT void ti_free_memory(TiDevice device, TiDeviceMemory deviceMemory);
TI_DLL_EXPORT void *ti_map_memory(TiDevice device, TiDeviceMemory deviceMemory);
TI_DLL_EXPORT void ti_unmap_memory(TiDevice device,
                                   TiDeviceMemory deviceMemory);

TI_DLL_EXPORT TiContext ti_create_context(TiDevice device);
TI_DLL_EXPORT void ti_destroy_context(TiContext context);

typedef struct TiNdShape {
  uint32_t dimCount;
  uint32_t dims[16];  // TODO: (penguinliong) give this constant a name?
} TiNdShape;
typedef struct TiNdArray {
  TiDeviceMemory devmem;
  TiNdShape shape;
  TiNdShape elem_shape;
} TiNdArray;
TI_DLL_EXPORT void ti_set_context_arg_ndarray(TiContext context,
                                              uint32_t argIndex,
                                              const TiNdArray *ndarray);
TI_DLL_EXPORT void ti_set_context_arg_i32(TiContext context,
                                          uint32_t argIndex,
                                          int32_t value);
TI_DLL_EXPORT void ti_set_context_arg_f32(TiContext context,
                                          uint32_t argIndex,
                                          float value);
TI_DLL_EXPORT void ti_launch_kernel(TiContext context, TiKernel kernel);

TI_DLL_EXPORT void ti_destroy_aot_module(TiAotModule mod);
TI_DLL_EXPORT TiKernel ti_get_aot_module_kernel(TiAotModule mod,
                                                const char *name);

#ifdef __cplusplus
}  // extern "C"
#endif
