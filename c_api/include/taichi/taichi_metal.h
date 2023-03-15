#pragma once

#ifndef TAICHI_H
#include "taichi.h"
#endif  // TAICHI_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Handle `TiNsBundle`
typedef struct TiNsBundle_t *TiNsBundle;

// Handle `TiMtlDevice`
typedef struct TiMtlDevice_t *TiMtlDevice;

// Handle `TiMtlBuffer`
typedef struct TiMtlBuffer_t *TiMtlBuffer;

// Handle `TiMtlTexture`
typedef struct TiMtlTexture_t *TiMtlTexture;

// Structure `TiMetalRuntimeInteropInfo`
typedef struct TiMetalRuntimeInteropInfo {
  TiNsBundle bundle;
  TiMtlDevice device;
} TiMetalRuntimeInteropInfo;

// Structure `TiMetalMemoryInteropInfo`
typedef struct TiMetalMemoryInteropInfo {
  TiMtlBuffer buffer;
} TiMetalMemoryInteropInfo;

// Structure `TiMetalImageInteropInfo`
typedef struct TiMetalImageInteropInfo {
  TiMtlTexture texture;
} TiMetalImageInteropInfo;

// Function `ti_import_metal_runtime`
TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_import_metal_runtime(const TiMetalRuntimeInteropInfo *interop_info);

// Function `ti_export_metal_runtime`
TI_DLL_EXPORT void TI_API_CALL
ti_export_metal_runtime(TiRuntime runtime,
                        TiMetalRuntimeInteropInfo *interop_info);

// Function `ti_import_metal_memory`
TI_DLL_EXPORT TiMemory TI_API_CALL
ti_import_metal_memory(TiRuntime runtime,
                       const TiMetalMemoryInteropInfo *interop_info);

// Function `ti_export_metal_memory`
TI_DLL_EXPORT void TI_API_CALL
ti_export_metal_memory(TiRuntime runtime,
                       TiMemory memory,
                       TiMetalMemoryInteropInfo *interop_info);

// Function `ti_import_metal_image`
TI_DLL_EXPORT TiImage TI_API_CALL
ti_import_metal_image(TiRuntime runtime,
                      const TiMetalImageInteropInfo *interop_info);

// Function `ti_export_metal_image`
TI_DLL_EXPORT void TI_API_CALL
ti_export_metal_image(TiRuntime runtime,
                      TiImage image,
                      TiMetalImageInteropInfo *interop_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
