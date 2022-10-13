#pragma once

#ifndef TI_WITH_CUDA
#define TI_WITH_CUDA 1
#endif  // TI_WITH_CUDA

#include <taichi/taichi.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// structure.cuda_memory_interop_info
typedef struct TiCudaMemoryInteropInfo {
  void *ptr;
  uint64_t size;
} TiCudaMemoryInteropInfo;

// function.export_cuda_memory
TI_DLL_EXPORT void TI_API_CALL
ti_export_cuda_memory(TiRuntime runtime,
                      TiMemory memory,
                      TiCudaMemoryInteropInfo *interop_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
