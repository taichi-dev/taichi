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
TI_DLL_EXPORT TiDevice TI_API_CALL ti_create_device(TiArch arch);
TI_DLL_EXPORT void TI_API_CALL ti_destroy_device(TiDevice device);
TI_DLL_EXPORT void TI_API_CALL ti_wait_device_idle(TiDevice device);

typedef enum TiAllocationUsageFlagBits {
  TI_MEMORY_USAGE_STORAGE_BIT = 1,
  TI_MEMORY_USAGE_UNIFORM_BIT = 2,
  TI_MEMORY_USAGE_VERTEX_BIT = 4,
  TI_MEMORY_USAGE_INDEX_BIT = 8,
} TiMemoryUsageFlagBits;
typedef TiFlags TiMemoryUsageFlags;
typedef struct {
  uint64_t size;
  TiBool host_write;
  TiBool host_read;
  TiBool export_sharing;
  TiMemoryUsageFlags usage;
} TiMemoryAllocateInfo;
TI_DLL_EXPORT TiDeviceMemory TI_API_CALL
ti_allocate_device_memory(TiDevice device,
                          const TiMemoryAllocateInfo *allocate_info);
TI_DLL_EXPORT void TI_API_CALL ti_free_device_memory(TiDevice device,
                                                     TiDeviceMemory devmem);
TI_DLL_EXPORT void *TI_API_CALL ti_map_device_memory(TiDevice device,
                                                     TiDeviceMemory devmem);
TI_DLL_EXPORT void TI_API_CALL ti_unmap_device_memory(TiDevice device,
                                                      TiDeviceMemory devmem);

TI_DLL_EXPORT TiContext TI_API_CALL ti_create_context(TiDevice device);
TI_DLL_EXPORT void TI_API_CALL ti_destroy_context(TiContext context);

typedef struct TiNdShape {
  uint32_t dim_count;
  uint32_t dims[16];  // TODO: (penguinliong) give this constant a name?
} TiNdShape;
typedef struct TiNdArray {
  TiDeviceMemory devmem;
  TiNdShape shape;
  TiNdShape elem_shape;
} TiNdArray;
TI_DLL_EXPORT void TI_API_CALL
ti_set_context_arg_ndarray(TiContext context,
                           uint32_t arg_index,
                           const TiNdArray *ndarray);
TI_DLL_EXPORT void TI_API_CALL ti_set_context_arg_i32(TiContext context,
                                                      uint32_t arg_index,
                                                      int32_t value);
TI_DLL_EXPORT void TI_API_CALL ti_set_context_arg_f32(TiContext context,
                                                      uint32_t arg_index,
                                                      float value);
TI_DLL_EXPORT void TI_API_CALL ti_launch_kernel(TiContext context,
                                                TiKernel kernel);
TI_DLL_EXPORT void TI_API_CALL ti_submit(TiContext context);
TI_DLL_EXPORT void TI_API_CALL ti_wait(TiContext context);

TI_DLL_EXPORT void TI_API_CALL ti_destroy_aot_module(TiAotModule mod);
TI_DLL_EXPORT TiKernel TI_API_CALL ti_get_aot_module_kernel(TiAotModule mod,
                                                            const char *name);

#ifdef __cplusplus
}  // extern "C"
#endif
