#pragma once

// Reference implementation:
// https://github.com/halide/Halide/blob/master/src/runtime/metal.cpp

#include <string>
#include <taichi/common.h>
#include <taichi/common/util.h>

#ifdef TC_SUPPORTS_METAL

#include <taichi/platform/mac/objc_api.h>

TLANG_NAMESPACE_BEGIN

namespace metal {

struct MTLDevice;
struct MTLLibrary;
struct MTLComputePipelineState;
struct MTLCommandQueue;
struct MTLCommandBuffer;
struct MTLComputeCommandEncoder;
struct MTLFunction;
struct MTLComputePipelineState;
struct MTLBuffer;

using mac::nsobj_unique_ptr;

nsobj_unique_ptr<MTLDevice> mtl_create_system_default_device();

nsobj_unique_ptr<MTLCommandQueue> new_command_queue(MTLDevice *dev);

nsobj_unique_ptr<MTLCommandBuffer> new_command_buffer(MTLCommandQueue *queue);

nsobj_unique_ptr<MTLComputeCommandEncoder>
new_compute_command_encoder(MTLCommandBuffer *buffer);

nsobj_unique_ptr<MTLLibrary> new_library_with_source(MTLDevice *device,
                                                     const std::string &source);

nsobj_unique_ptr<MTLFunction> new_function_with_name(MTLLibrary *library,
                                                     const std::string &name);

nsobj_unique_ptr<MTLComputePipelineState>
new_compute_pipeline_state_with_function(MTLDevice *device,
                                         MTLFunction *function);

inline void
set_compute_pipeline_state(MTLComputeCommandEncoder *encoder,
                           MTLComputePipelineState *pipeline_state) {
  mac::call(encoder, "setComputePipelineState:", pipeline_state);
}

inline void end_encoding(MTLComputeCommandEncoder *encoder) {
  mac::call(encoder, "endEncoding");
}

nsobj_unique_ptr<MTLBuffer> new_mtl_buffer(MTLDevice *device, size_t length);

nsobj_unique_ptr<MTLBuffer> new_mtl_buffer_no_copy(MTLDevice *device, void *ptr,
                                                   size_t length);

inline void set_mtl_buffer(MTLComputeCommandEncoder *encoder, MTLBuffer *buffer,
                           size_t offset, size_t index) {
  mac::call(encoder, "setBuffer:offset:atIndex:", buffer, offset, index);
}

void dispatch_threadgroups(MTLComputeCommandEncoder *encoder, int32_t blocks_x,
                           int32_t blocks_y, int32_t blocks_z,
                           int32_t threads_x, int32_t threads_y,
                           int32_t threads_z);

// 1D
inline void dispatch_threadgroups(MTLComputeCommandEncoder *encoder,
                                  int32_t blocks_x, int32_t threads_x) {
  dispatch_threadgroups(encoder, blocks_x, 1, 1, threads_x, 1, 1);
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

} // namespace metal

TLANG_NAMESPACE_END

#endif  // TC_SUPPORTS_METAL
