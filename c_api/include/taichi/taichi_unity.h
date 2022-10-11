#pragma once

#include <taichi/taichi.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// handle.native_buffer
typedef struct TixNativeBufferUnity_t *TixNativeBufferUnity;

// function.import_native_runtime
TI_DLL_EXPORT TiRuntime TI_API_CALL tix_import_native_runtime_unity();

// function.launch_kernel_async
TI_DLL_EXPORT void TI_API_CALL
tix_launch_kernel_async_unity(TiRuntime runtime,
                              TiKernel kernel,
                              uint32_t arg_count,
                              const TiArgument *args);

// function.launch_compute_graph_async
TI_DLL_EXPORT void TI_API_CALL
tix_launch_compute_graph_async_unity(TiRuntime runtime,
                                     TiComputeGraph compute_graph,
                                     uint32_t arg_count,
                                     const TiNamedArgument *args);

// function.copy_memory_to_native_buffer_async
TI_DLL_EXPORT void TI_API_CALL
tix_copy_memory_to_native_buffer_async_unity(TiRuntime runtime,
                                             TixNativeBufferUnity dst,
                                             uint64_t dst_offset,
                                             const TiMemorySlice *src);

// function.copy_memory_device_to_host
TI_DLL_EXPORT void TI_API_CALL
tix_copy_memory_device_to_host_unity(TiRuntime runtime,
                                     void *dst,
                                     uint64_t dst_offset,
                                     const TiMemorySlice *src);

// function.copy_memory_host_to_device
TI_DLL_EXPORT void TI_API_CALL
tix_copy_memory_host_to_device_unity(TiRuntime runtime,
                                     const TiMemorySlice *dst,
                                     const void *src,
                                     uint64_t src_offset);

// function.submit_async
TI_DLL_EXPORT void *TI_API_CALL tix_submit_async_unity(TiRuntime runtime);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
