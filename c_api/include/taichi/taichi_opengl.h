#pragma once

#include <taichi/taichi.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Structure `TiOpenglRuntimeInteropInfo`
typedef struct TiOpenglRuntimeInteropInfo {
  void *get_proc_addr;
} TiOpenglRuntimeInteropInfo;

// Function `ti_import_opengl_runtime`
TI_DLL_EXPORT void TI_API_CALL
ti_import_opengl_runtime(TiRuntime runtime,
                         TiOpenglRuntimeInteropInfo *interop_info);

// Function `ti_export_opengl_runtime`
TI_DLL_EXPORT void TI_API_CALL
ti_export_opengl_runtime(TiRuntime runtime,
                         TiOpenglRuntimeInteropInfo *interop_info);

// Structure `TiOpenglMemoryInteropInfo`
typedef struct TiOpenglMemoryInteropInfo {
  GLuint buffer;
  uint64_t size;
} TiOpenglMemoryInteropInfo;

// Function `ti_import_opengl_memory`
TI_DLL_EXPORT void TI_API_CALL
ti_import_opengl_memory(TiRuntime runtime,
                        TiMemory memory,
                        TiOpenglMemoryInteropInfo *interop_info);

// Function `ti_export_opengl_memory`
TI_DLL_EXPORT void TI_API_CALL
ti_export_opengl_memory(TiRuntime runtime,
                        TiMemory memory,
                        TiOpenglMemoryInteropInfo *interop_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
