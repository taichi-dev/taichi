#include <taichi/taichi.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

TI_DLL_EXPORT void TI_API_CALL ti_get_available_archs(uint32_t *arch_count,
                                                      TiArch *archs) {
}
TI_DLL_EXPORT TiError TI_API_CALL ti_get_last_error(uint64_t message_size,
                                                    char *message) {
  return (TiError)0;
}
TI_DLL_EXPORT void TI_API_CALL ti_set_last_error(TiError error,
                                                 const char *message) {
}
TI_DLL_EXPORT TiRuntime TI_API_CALL ti_create_runtime(TiArch arch) {
  return (TiRuntime)0;
}
TI_DLL_EXPORT void TI_API_CALL ti_destroy_runtime(TiRuntime runtime) {
}
TI_DLL_EXPORT void TI_API_CALL
ti_get_runtime_capabilities(TiRuntime runtime,
                            uint32_t *capability_count,
                            TiCapabilityLevelInfo *capabilities) {
}
TI_DLL_EXPORT TiMemory TI_API_CALL
ti_allocate_memory(TiRuntime runtime,
                   const TiMemoryAllocateInfo *allocate_info) {
  return (TiMemory)0;
}
TI_DLL_EXPORT void TI_API_CALL ti_free_memory(TiRuntime runtime,
                                              TiMemory memory) {
}
TI_DLL_EXPORT void *TI_API_CALL ti_map_memory(TiRuntime runtime,
                                              TiMemory memory) {
  return (void *)0;
}
TI_DLL_EXPORT void TI_API_CALL ti_unmap_memory(TiRuntime runtime,
                                               TiMemory memory) {
}
TI_DLL_EXPORT TiImage TI_API_CALL
ti_allocate_image(TiRuntime runtime, const TiImageAllocateInfo *allocate_info) {
  return (TiImage)0;
}
TI_DLL_EXPORT void TI_API_CALL ti_free_image(TiRuntime runtime, TiImage image) {
}
TI_DLL_EXPORT TiSampler TI_API_CALL
ti_create_sampler(TiRuntime runtime, const TiSamplerCreateInfo *create_info) {
  return (TiSampler)0;
}
TI_DLL_EXPORT void TI_API_CALL ti_destroy_sampler(TiRuntime runtime,
                                                  TiSampler sampler) {
}
TI_DLL_EXPORT TiEvent TI_API_CALL ti_create_event(TiRuntime runtime) {
  return (TiEvent)0;
}
TI_DLL_EXPORT void TI_API_CALL ti_destroy_event(TiEvent event) {
}
TI_DLL_EXPORT void TI_API_CALL
ti_copy_memory_device_to_device(TiRuntime runtime,
                                const TiMemorySlice *dst_memory,
                                const TiMemorySlice *src_memory) {
}
TI_DLL_EXPORT void TI_API_CALL
ti_copy_image_device_to_device(TiRuntime runtime,
                               const TiImageSlice *dst_image,
                               const TiImageSlice *src_image) {
}
TI_DLL_EXPORT void TI_API_CALL ti_track_image_ext(TiRuntime runtime,
                                                  TiImage image,
                                                  TiImageLayout layout) {
}
TI_DLL_EXPORT void TI_API_CALL ti_transition_image(TiRuntime runtime,
                                                   TiImage image,
                                                   TiImageLayout layout) {
}
TI_DLL_EXPORT void TI_API_CALL ti_launch_kernel(TiRuntime runtime,
                                                TiKernel kernel,
                                                uint32_t arg_count,
                                                const TiArgument *args) {
}
TI_DLL_EXPORT void TI_API_CALL
ti_launch_compute_graph(TiRuntime runtime,
                        TiComputeGraph compute_graph,
                        uint32_t arg_count,
                        const TiNamedArgument *args) {
}
TI_DLL_EXPORT void TI_API_CALL ti_signal_event(TiRuntime runtime,
                                               TiEvent event) {
}
TI_DLL_EXPORT void TI_API_CALL ti_reset_event(TiRuntime runtime,
                                              TiEvent event) {
}
TI_DLL_EXPORT void TI_API_CALL ti_wait_event(TiRuntime runtime, TiEvent event) {
}
TI_DLL_EXPORT void TI_API_CALL ti_submit(TiRuntime runtime) {
}
TI_DLL_EXPORT void TI_API_CALL ti_wait(TiRuntime runtime) {
}
TI_DLL_EXPORT TiAotModule TI_API_CALL
ti_load_aot_module(TiRuntime runtime, const char *module_path) {
  return (TiAotModule)0;
}
TI_DLL_EXPORT TiAotModule TI_API_CALL ti_create_aot_module(TiRuntime runtime,
                                                           const void *tcm,
                                                           uint64_t size) {
  return (TiAotModule)0;
}
TI_DLL_EXPORT void TI_API_CALL ti_destroy_aot_module(TiAotModule aot_module) {
}
TI_DLL_EXPORT TiKernel TI_API_CALL
ti_get_aot_module_kernel(TiAotModule aot_module, const char *name) {
  return (TiKernel)0;
}
TI_DLL_EXPORT TiComputeGraph TI_API_CALL
ti_get_aot_module_compute_graph(TiAotModule aot_module, const char *name) {
  return (TiComputeGraph)0;
}
#ifdef TI_WITH_CPU

TI_DLL_EXPORT void TI_API_CALL
ti_export_cpu_memory(TiRuntime runtime,
                     TiMemory memory,
                     TiCpuMemoryInteropInfo *interop_info) {
}
#endif  // TI_WITH_CPU

#ifdef TI_WITH_CUDA

TI_DLL_EXPORT void TI_API_CALL
ti_export_cuda_memory(TiRuntime runtime,
                      TiMemory memory,
                      TiCudaMemoryInteropInfo *interop_info) {
}
#endif  // TI_WITH_CUDA

#ifdef TI_WITH_VULKAN

TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_create_vulkan_runtime_ext(uint32_t api_version,
                             uint32_t instance_extension_count,
                             const char **instance_extensions,
                             uint32_t device_extension_count,
                             const char **device_extensions) {
  return (TiRuntime)0;
}
TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_import_vulkan_runtime(const TiVulkanRuntimeInteropInfo *interop_info) {
  return (TiRuntime)0;
}
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_runtime(TiRuntime runtime,
                         TiVulkanRuntimeInteropInfo *interop_info) {
}
TI_DLL_EXPORT TiMemory TI_API_CALL
ti_import_vulkan_memory(TiRuntime runtime,
                        const TiVulkanMemoryInteropInfo *interop_info) {
  return (TiMemory)0;
}
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_memory(TiRuntime runtime,
                        TiMemory memory,
                        TiVulkanMemoryInteropInfo *interop_info) {
}
TI_DLL_EXPORT TiImage TI_API_CALL
ti_import_vulkan_image(TiRuntime runtime,
                       const TiVulkanImageInteropInfo *interop_info,
                       VkImageViewType view_type,
                       VkImageLayout layout) {
  return (TiImage)0;
}
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_image(TiRuntime runtime,
                       TiImage image,
                       TiVulkanImageInteropInfo *interop_info) {
}
TI_DLL_EXPORT TiEvent TI_API_CALL
ti_import_vulkan_event(TiRuntime runtime,
                       const TiVulkanEventInteropInfo *interop_info) {
  return (TiEvent)0;
}
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_event(TiRuntime runtime,
                       TiEvent event,
                       TiVulkanEventInteropInfo *interop_info) {
}
#endif  // TI_WITH_VULKAN

#ifdef TI_WITH_OPENGL

TI_DLL_EXPORT void TI_API_CALL
ti_import_opengl_memory(TiRuntime runtime,
                        TiMemory memory,
                        TiOpenglMemoryInteropInfo *interop_info) {
}
TI_DLL_EXPORT void TI_API_CALL
ti_export_opengl_memory(TiRuntime runtime,
                        TiMemory memory,
                        TiOpenglMemoryInteropInfo *interop_info) {
}
#endif  // TI_WITH_OPENGL

#ifdef TI_WITH_UNITY

TI_DLL_EXPORT TiRuntime TI_API_CALL tix_import_native_runtime_unity() {
  return (TiRuntime)0;
}
TI_DLL_EXPORT void TI_API_CALL
tix_launch_kernel_async_unity(TiRuntime runtime,
                              TiKernel kernel,
                              uint32_t arg_count,
                              const TiArgument *args) {
}
TI_DLL_EXPORT void TI_API_CALL
tix_launch_compute_graph_async_unity(TiRuntime runtime,
                                     TiComputeGraph compute_graph,
                                     uint32_t arg_count,
                                     const TiNamedArgument *args) {
}
TI_DLL_EXPORT void TI_API_CALL
tix_copy_memory_to_native_buffer_async_unity(TiRuntime runtime,
                                             TixNativeBufferUnity dst,
                                             uint64_t dst_offset,
                                             const TiMemorySlice *src) {
}
TI_DLL_EXPORT void TI_API_CALL
tix_copy_memory_device_to_host_unity(TiRuntime runtime,
                                     void *dst,
                                     uint64_t dst_offset,
                                     const TiMemorySlice *src) {
}
TI_DLL_EXPORT void TI_API_CALL
tix_copy_memory_host_to_device_unity(TiRuntime runtime,
                                     const TiMemorySlice *dst,
                                     const void *src,
                                     uint64_t src_offset) {
}
TI_DLL_EXPORT void *TI_API_CALL tix_submit_async_unity(TiRuntime runtime) {
  return (void *)0;
}
#endif  // TI_WITH_UNITY

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
