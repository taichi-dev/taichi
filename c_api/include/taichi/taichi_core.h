#pragma once

#include "taichi/taichi_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t TiBool;

typedef void* TiOpaqueHandle;


typedef TiOpaqueHandle TiContext;
typedef TiOpaqueHandle TiKernel;
typedef TiOpaqueHandle TiAotModule;
typedef TiOpaqueHandle TiDeviceAllocation;

TI_DLL_EXPORT TiContext tiCreateContext();
TI_DLL_EXPORT void tiDestroyContext(TiContext ctx);



TI_DLL_EXPORT void tiSetContextArgumentI32(TiContext context,
                                           uint32_t argumentIndex,
                                           int32_t value);
TI_DLL_EXPORT void tiSetContextArgumentF32(TiContext context,
                                           uint32_t argumentIndex,
                                           float value);
typedef struct TiNdShape {
  uint32_t data[16];  // TODO: (penguinliong) give this constant a name?
  uint32_t length;
} TiNdShape;
typedef struct TiNdArray {
  TiDeviceAllocation devalloc;
  const TiNdShape *shape;
  const TiNdShape *elem_shape;
} TiNdArray;
TI_DLL_EXPORT void tiSetContextArgumentNdArray(TiContext context,
                                               uint32_t paramIndex,
                                               const TiNdArray* arr);
TI_DLL_EXPORT void tiLaunchKernel(TiContext ctx, TiKernel k);



TI_DLL_EXPORT TiKernel tiGetAotModuleKernel(
    TiAotModule *m,
    const char *name);
TI_DLL_EXPORT size_t tiGetAotModuleRootBufferSize(TiAotModule *m);



typedef TiOpaqueHandle TiDevice;
TI_DLL_EXPORT void tiDeviceWaitIdle(TiDevice device);
TI_DLL_EXPORT void tiDestroyDevice(TiDevice device);



typedef struct {
  uint64_t size;
  TiBool hostWritable;
  TiBool hostReadable;
  TiBool exportSharing;
  // AllocUsage is an enum class, so not exported to C yet
} TiDeviceAllocationInfo;
TI_DLL_EXPORT TiDeviceAllocation *tiCreateDeviceAllocation(
  TiDevice dev,
  const TiDeviceAllocationInfo *params);
TI_DLL_EXPORT void tiDestroyDeviceAllocation(
  TiDevice dev,
  TiDeviceAllocation da);
TI_DLL_EXPORT void *tiMapDeviceAllocation(
  TiDevice dev,
  TiDeviceAllocation da);
TI_DLL_EXPORT void tiUnmapDeviceAllocation(
  TiDevice dev,
  TiDeviceAllocation da);



#ifdef __cplusplus
}  // extern "C"
#endif
