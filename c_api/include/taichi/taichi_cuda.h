#pragma once
#include <taichi/taichi_core.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/*
    We want to avoid including the entire <cuda.h> in C-API headers,
    which exposes binary compatibility risks in the scenario where
    the <cuda.h> used by C-API users is different from that used during taichi
   compilation.

    Thus we only copy neccessary declarations from <cuda.h>
*/
typedef void
    *CUdeviceptr;  // Same implementation as "unsigned integer type matching
                   // size of the pointer on target platform" in <cuda.h>

// structure.cuda_memory_interop_info
typedef struct TiCudaMemoryInteropInfo {
  CUdeviceptr ptr;
  uint64_t size;
} TiCudaMemoryInteropInfo;

// function.export_cuda_runtime
TI_DLL_EXPORT void TI_API_CALL
ti_export_cuda_memory(TiRuntime runtime,
                      TiMemory memory,
                      TiCudaMemoryInteropInfo *interop_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
