#pragma once

#ifndef TAICHI_H
#include "taichi.h"
#endif  // TAICHI_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Structure `TiCudaMemoryInteropInfo`
typedef struct TiCudaMemoryInteropInfo {
  void *ptr;
  uint64_t size;
} TiCudaMemoryInteropInfo;

// Function `ti_export_cuda_memory`
TI_DLL_EXPORT void TI_API_CALL
ti_export_cuda_memory(TiRuntime runtime,
                      TiMemory memory,
                      TiCudaMemoryInteropInfo *interop_info);

// Function `ti_import_cuda_memory`
TI_DLL_EXPORT TiMemory TI_API_CALL ti_import_cuda_memory(TiRuntime runtime,
                                                         void *ptr,
                                                         size_t memory_size);

// Function `ti_set_cuda_stream`
TI_DLL_EXPORT void TI_API_CALL ti_set_cuda_stream(void *stream);

// Function `ti_get_cuda_stream`
TI_DLL_EXPORT void TI_API_CALL ti_get_cuda_stream(void **stream);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
