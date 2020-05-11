#pragma once

// Reference implementation:
// https://github.com/halide/Halide/blob/master/src/runtime/metal.cpp

#include "taichi/lang_util.h"
#include "taichi/platform/mac/objc_api.h"

#include <string>

TLANG_NAMESPACE_BEGIN

namespace metal {

// Expose these incomplete structs so that other modules (e.g. MetalRuntime)
// don't have to be compiled conditionally.
struct MTLDevice;
struct MTLLibrary;
struct MTLComputePipelineState;
struct MTLCommandQueue;
struct MTLCommandBuffer;
struct MTLComputeCommandEncoder;
struct MTLFunction;
struct MTLComputePipelineState;
struct MTLBuffer;

#ifdef TI_PLATFORM_OSX

using mac::nsobj_unique_ptr;

nsobj_unique_ptr<MTLDevice> mtl_create_system_default_device();

nsobj_unique_ptr<MTLCommandQueue> new_command_queue(MTLDevice *dev);

nsobj_unique_ptr<MTLCommandBuffer> new_command_buffer(MTLCommandQueue *queue);

nsobj_unique_ptr<MTLComputeCommandEncoder> new_compute_command_encoder(
    MTLCommandBuffer *buffer);

nsobj_unique_ptr<MTLLibrary> new_library_with_source(MTLDevice *device,
                                                     const std::string &source);

nsobj_unique_ptr<MTLFunction> new_function_with_name(MTLLibrary *library,
                                                     const std::string &name);

nsobj_unique_ptr<MTLComputePipelineState>
new_compute_pipeline_state_with_function(MTLDevice *device,
                                         MTLFunction *function);

inline void set_compute_pipeline_state(
    MTLComputeCommandEncoder *encoder,
    MTLComputePipelineState *pipeline_state) {
  mac::call(encoder, "setComputePipelineState:", pipeline_state);
}

inline void end_encoding(MTLComputeCommandEncoder *encoder) {
  mac::call(encoder, "endEncoding");
}

nsobj_unique_ptr<MTLBuffer> new_mtl_buffer(MTLDevice *device, size_t length);

nsobj_unique_ptr<MTLBuffer> new_mtl_buffer_no_copy(MTLDevice *device,
                                                   void *ptr,
                                                   size_t length);

inline void set_mtl_buffer(MTLComputeCommandEncoder *encoder,
                           MTLBuffer *buffer,
                           size_t offset,
                           size_t index) {
  mac::call(encoder, "setBuffer:offset:atIndex:", buffer, offset, index);
}

void dispatch_threadgroups(MTLComputeCommandEncoder *encoder,
                           int32_t blocks_x,
                           int32_t blocks_y,
                           int32_t blocks_z,
                           int32_t threads_x,
                           int32_t threads_y,
                           int32_t threads_z);

// 1D
inline void dispatch_threadgroups(MTLComputeCommandEncoder *encoder,
                                  int32_t blocks_x,
                                  int32_t threads_x) {
  dispatch_threadgroups(encoder, blocks_x, 1, 1, threads_x, 1, 1);
}

template <
    typename T,
    typename = std::enable_if_t<std::is_same_v<T, MTLComputeCommandEncoder> ||
                                std::is_same_v<T, MTLCommandBuffer>>>
void set_label(T *mtl_obj, const std::string &label) {
  // Set labels on Metal command buffer and encoders, so that they can be
  // tracked in Instrument - Metal System Trace
  auto label_str = mac::wrap_string_as_ns_string(label);
  mac::call(mtl_obj, "setLabel:", label_str.get());
}

inline void enqueue_command_buffer(MTLCommandBuffer *cmd_buffer) {
  mac::call(cmd_buffer, "enqueue");
}

inline void commit_command_buffer(MTLCommandBuffer *cmd_buffer) {
  mac::call(cmd_buffer, "commit");
}

inline void wait_until_completed(MTLCommandBuffer *cmd_buffer) {
  mac::call(cmd_buffer, "waitUntilCompleted");
}

inline void *mtl_buffer_contents(MTLBuffer *buffer) {
  return mac::cast_call<void *>(buffer, "contents");
}

size_t get_max_total_threads_per_threadgroup(
    MTLComputePipelineState *pipeline_state);
#endif  // TI_PLATFORM_OSX

bool is_metal_api_available();

}  // namespace metal

TLANG_NAMESPACE_END
