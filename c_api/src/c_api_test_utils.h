#pragma once
#include <taichi/taichi_platform.h>
#include "taichi/taichi_core.h"

namespace capi {
namespace utils {

TI_DLL_EXPORT bool TI_API_CALL is_vulkan_available();
TI_DLL_EXPORT bool TI_API_CALL is_opengl_available();
TI_DLL_EXPORT bool TI_API_CALL is_cuda_available();
TI_DLL_EXPORT void TI_API_CALL check_runtime_error(TiRuntime runtime);

typedef struct TiNdarrayAndMem {
  TiRuntime runtime_;
  TiMemory memory_;
  TiArgument arg_;
} TiNdarrayAndMem;

TI_DLL_EXPORT TiNdarrayAndMem TI_API_CALL make_ndarray(TiRuntime runtime,
                                                       TiDataType dtype,
                                                       const int *arr_shape,
                                                       int arr_dims,
                                                       const int *element_shape,
                                                       int element_dims,
                                                       bool host_read = false,
                                                       bool host_write = false);

}  // namespace utils
}  // namespace capi
