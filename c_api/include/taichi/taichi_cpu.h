#pragma once

#ifndef TAICHI_H
#include "taichi.h"
#endif  // TAICHI_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Structure `TiCpuMemoryInteropInfo`
typedef struct TiCpuMemoryInteropInfo {
  void *ptr;
  uint64_t size;
} TiCpuMemoryInteropInfo;

// Function `ti_export_cpu_memory`
TI_DLL_EXPORT void TI_API_CALL
ti_export_cpu_memory(TiRuntime runtime,
                     TiMemory memory,
                     TiCpuMemoryInteropInfo *interop_info);

TI_DLL_EXPORT TiMemory TI_API_CALL ti_import_cpu_memory(TiRuntime runtime,
                                                        void *ptr,
                                                        size_t memory_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
