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

TI_DLL_EXPORT TiMemory TI_API_CALL ti_import_cuda_memory(TiRuntime runtime,
                                                         void *ptr,
                                                         size_t memory_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
