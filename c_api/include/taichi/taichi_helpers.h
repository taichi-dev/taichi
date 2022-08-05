#pragma once
#include <taichi/taichi_core.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
