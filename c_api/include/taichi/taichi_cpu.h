#pragma once

#include <taichi/taichi.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// structure.cpu_memory_interop_info
typedef struct TiCpuMemoryInteropInfo {
  void *ptr;
  uint64_t size;
} TiCpuMemoryInteropInfo;

// function.export_cpu_memory
TI_DLL_EXPORT void TI_API_CALL
ti_export_cpu_memory(TiRuntime runtime,
                     TiMemory memory,
                     TiCpuMemoryInteropInfo *interop_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
