#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "taichi/common/platform_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TaichiKernel TaichiKernel;
typedef struct TaichiRuntimeContext TaichiRuntimeContext;
typedef struct DeviceAllocation DeviceAllocation;

typedef TI_DLL_EXPORT struct {
  size_t size;
  bool host_write;
  bool host_read;
  bool export_sharing;
  // AllocUsage is an enum class, so not exported to C yet
} DeviceAllocParams;

TI_DLL_EXPORT void launch_taichi_kernel(TaichiKernel *k,
                                        TaichiRuntimeContext *ctx);

TI_DLL_EXPORT TaichiRuntimeContext *make_runtime_context();

TI_DLL_EXPORT void destroy_runtime_context(TaichiRuntimeContext *ctx);

TI_DLL_EXPORT void set_runtime_context_arg_i32(TaichiRuntimeContext *ctx,
                                               int param_i, int32_t val);

TI_DLL_EXPORT void set_runtime_context_arg_float(TaichiRuntimeContext *ctx,
                                                 int param_i, float val);

typedef TI_DLL_EXPORT struct {
  int32_t length;
  int32_t data[1];
} NdShape;

TI_DLL_EXPORT void set_runtime_context_arg_ndarray(TaichiRuntimeContext *ctx,
                                                   int param_i,
                                                   DeviceAllocation *arr,
                                                   const NdShape *shape,
                                                   const NdShape *elem_shape);

TI_DLL_EXPORT void set_runtime_context_arg_scalar_ndarray(
    TaichiRuntimeContext *ctx, int param_i, DeviceAllocation *arr,
    const NdShape *shape);

#ifdef __cplusplus
}  // extern "C"
#endif
