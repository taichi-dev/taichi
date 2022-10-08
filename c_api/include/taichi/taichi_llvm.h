#pragma once
#include <taichi/taichi_core.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// structure.cpu_memory_interop_info
typedef struct TiCpuMemoryInteropInfo {
  void *ptr;
  uint64_t size;
} TiCpuMemoryInteropInfo;

// structure.cuda_memory_interop_info
typedef struct TiCudaMemoryInteropInfo {
  void *ptr;
  uint64_t size;
} TiCudaMemoryInteropInfo;

// function.export_cpu_runtime
TI_DLL_EXPORT void TI_API_CALL
ti_export_cpu_runtime(TiRuntime runtime,
                      TiMemory memory,
                      TiCpuMemoryInteropInfo *interop_info);

// function.export_cuda_runtime
TI_DLL_EXPORT void TI_API_CALL
ti_export_cuda_runtime(TiRuntime runtime,
                       TiMemory memory,
                       TiCudaMemoryInteropInfo *interop_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
