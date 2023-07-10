#pragma once

#ifndef TAICHI_H
#include "taichi.h"
#endif  // TAICHI_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Structure `TiOpenglRuntimeInteropInfo`
typedef struct TiOpenglRuntimeInteropInfo {
  void *get_proc_addr;
} TiOpenglRuntimeInteropInfo;

// Structure `TiOpenglMemoryInteropInfo`
typedef struct TiOpenglMemoryInteropInfo {
  GLuint buffer;
  GLsizeiptr size;
} TiOpenglMemoryInteropInfo;

// Structure `TiOpenglImageInteropInfo`
typedef struct TiOpenglImageInteropInfo {
  GLuint texture;
  GLenum target;
  GLsizei levels;
  GLenum format;
  GLsizei width;
  GLsizei height;
  GLsizei depth;
} TiOpenglImageInteropInfo;

// Function `ti_import_opengl_runtime`
TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_import_opengl_runtime(TiOpenglRuntimeInteropInfo *interop_info,
                         bool use_gles);

// Function `ti_export_opengl_runtime`
TI_DLL_EXPORT void TI_API_CALL
ti_export_opengl_runtime(TiRuntime runtime,
                         TiOpenglRuntimeInteropInfo *interop_info);

// Function `ti_import_opengl_memory`
TI_DLL_EXPORT TiMemory TI_API_CALL
ti_import_opengl_memory(TiRuntime runtime,
                        const TiOpenglMemoryInteropInfo *interop_info);

// Function `ti_export_opengl_memory`
TI_DLL_EXPORT void TI_API_CALL
ti_export_opengl_memory(TiRuntime runtime,
                        TiMemory memory,
                        TiOpenglMemoryInteropInfo *interop_info);

// Function `ti_import_opengl_image`
TI_DLL_EXPORT TiImage TI_API_CALL
ti_import_opengl_image(TiRuntime runtime,
                       const TiOpenglImageInteropInfo *interop_info);

// Function `ti_export_opengl_image`
TI_DLL_EXPORT void TI_API_CALL
ti_export_opengl_image(TiRuntime runtime,
                       TiImage image,
                       TiOpenglImageInteropInfo *interop_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
