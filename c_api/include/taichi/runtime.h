#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "c_api/include/taichi/backends/device_api.h"
#include "taichi/common/platform_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Taichi_Kernel Taichi_Kernel;
typedef struct TaichTaichi_RuntimeContextiRuntimeContext Taichi_RuntimeContext;

TI_DLL_EXPORT void taichi_launch_kernel(Taichi_Kernel *k,
                                        Taichi_RuntimeContext *ctx);

TI_DLL_EXPORT Taichi_RuntimeContext *taichi_make_runtime_context();

TI_DLL_EXPORT void taichi_destroy_runtime_context(Taichi_RuntimeContext *ctx);

TI_DLL_EXPORT void taichi_set_runtime_context_arg_i32(
    Taichi_RuntimeContext *ctx, int param_i, int32_t val);

TI_DLL_EXPORT void taichi_set_runtime_context_arg_float(
    Taichi_RuntimeContext *ctx, int param_i, float val);

typedef TI_DLL_EXPORT struct {
  int32_t length;
  int32_t data[1];
} Taichi_NdShape;

TI_DLL_EXPORT void taichi_set_runtime_context_arg_ndarray(
    Taichi_RuntimeContext *ctx, int param_i, Taichi_DeviceAllocation *arr,
    const Taichi_NdShape *shape, const Taichi_NdShape *elem_shape);

TI_DLL_EXPORT void taichi_set_runtime_context_arg_scalar_ndarray(
    Taichi_RuntimeContext *ctx, int param_i, Taichi_DeviceAllocation *arr,
    const Taichi_NdShape *shape);

#ifdef __cplusplus
}  // extern "C"
#endif
