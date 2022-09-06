#pragma once
#include <taichi/taichi_core.h>
#include <glad/gl.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// structure.opengl_memory_interop_info
typedef struct TiOpenglMemoryInteropInfo {
  GLuint buffer;
  uint64_t size;
} TiOpenglMemoryInteropInfo;

// function.export_opengl_memory
TI_DLL_EXPORT void TI_API_CALL
ti_export_opengl_memory(TiRuntime runtime,
                        TiMemory memory,
                        TiOpenglMemoryInteropInfo *interop_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
