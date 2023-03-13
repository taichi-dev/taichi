#pragma once

#ifndef TAICHI_H
#include "taichi.h"
#endif  // TAICHI_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Handle `TixNativeBufferUnity`
typedef struct TixNativeBufferUnity_t *TixNativeBufferUnity;

// Callback `TixAsyncTaskUnity`
typedef void(TI_API_CALL *TixAsyncTaskUnity)(void *user_data);

// Function `tix_import_native_runtime_unity`
TI_DLL_EXPORT TiRuntime TI_API_CALL tix_import_native_runtime_unity();

// Function `tix_enqueue_task_async_unity`
TI_DLL_EXPORT void TI_API_CALL
tix_enqueue_task_async_unity(void *user_data, TixAsyncTaskUnity async_task);

// Function `tix_launch_kernel_async_unity`
TI_DLL_EXPORT void TI_API_CALL
tix_launch_kernel_async_unity(TiRuntime runtime,
                              TiKernel kernel,
                              uint32_t arg_count,
                              const TiArgument *args);

// Function `tix_launch_compute_graph_async_unity`
TI_DLL_EXPORT void TI_API_CALL
tix_launch_compute_graph_async_unity(TiRuntime runtime,
                                     TiComputeGraph compute_graph,
                                     uint32_t arg_count,
                                     const TiNamedArgument *args);

// Function `tix_copy_memory_to_native_buffer_async_unity`
TI_DLL_EXPORT void TI_API_CALL
tix_copy_memory_to_native_buffer_async_unity(TiRuntime runtime,
                                             TixNativeBufferUnity dst,
                                             uint64_t dst_offset,
                                             const TiMemorySlice *src);

// Function `tix_copy_memory_device_to_host_unity`
TI_DLL_EXPORT void TI_API_CALL
tix_copy_memory_device_to_host_unity(TiRuntime runtime,
                                     void *dst,
                                     uint64_t dst_offset,
                                     const TiMemorySlice *src);

// Function `tix_copy_memory_host_to_device_unity`
TI_DLL_EXPORT void TI_API_CALL
tix_copy_memory_host_to_device_unity(TiRuntime runtime,
                                     const TiMemorySlice *dst,
                                     const void *src,
                                     uint64_t src_offset);

// Function `tix_submit_async_unity`
TI_DLL_EXPORT void *TI_API_CALL tix_submit_async_unity(TiRuntime runtime);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
