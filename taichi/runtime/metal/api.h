#pragma once

// Reference implementation:
// https://github.com/halide/Halide/blob/master/src/runtime/metal.cpp

#include <string>

#include "taichi/common/trait.h"
#include "taichi/lang_util.h"
#include "taichi/platform/mac/objc_api.h"

namespace taichi {
namespace lang {
namespace metal {

// Expose these incomplete structs so that other modules (e.g. MetalRuntime)
// don't have to be compiled conditionally.
struct MTLDevice;
struct MTLLibrary;
struct MTLComputePipelineState;
struct MTLCommandQueue;
struct MTLCommandBuffer;
struct MTLComputeCommandEncoder;
struct MTLBlitCommandEncoder;
struct MTLFunction;
struct MTLComputePipelineState;
struct MTLBuffer;

#ifdef TI_PLATFORM_OSX

using mac::nsobj_unique_ptr;

nsobj_unique_ptr<MTLDevice> mtl_create_system_default_device();

nsobj_unique_ptr<mac::TI_NSArray> mtl_copy_all_devices();

std::string mtl_device_name(MTLDevice *dev);

nsobj_unique_ptr<MTLCommandQueue> new_command_queue(MTLDevice *dev);

nsobj_unique_ptr<MTLCommandBuffer> new_command_buffer(MTLCommandQueue *queue);

nsobj_unique_ptr<MTLComputeCommandEncoder> new_compute_command_encoder(
    MTLCommandBuffer *buffer);

nsobj_unique_ptr<MTLBlitCommandEncoder> new_blit_command_encoder(
    MTLCommandBuffer *buffer);

// msl_version: Metal Shader Language version. 0 means not set.
// See https://developer.apple.com/documentation/metal/mtllanguageversion
nsobj_unique_ptr<MTLLibrary> new_library_with_source(MTLDevice *device,
                                                     const std::string &source,
                                                     bool fast_math,
                                                     int msl_version);

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

inline void synchronize_resource(MTLBlitCommandEncoder *encoder,
                                 MTLBuffer *buffer) {
  mac::call(encoder, "synchronizeResource:", buffer);
}

template <typename T>
inline void end_encoding(T *encoder) {
  static_assert(std::is_same_v<T, MTLComputeCommandEncoder> ||
                std::is_same_v<T, MTLBlitCommandEncoder>);
  mac::call(encoder, "endEncoding");
}

// The created MTLBuffer has its storege mode being .manged.
// API ref:
// https://developer.apple.com/documentation/metal/mtldevice/1433382-makebuffer
//
// We initially used .shared storage mode, meaning the GPU and CPU shared the
// system memory. This turned out to be slow as page fault on GPU was very
// costly. By switching to .managed mode, on GPUs with discrete memory model,
// the data will reside in both GPU's VRAM and CPU's system RAM. This made the
// GPU memory access much faster. But we will need to manually synchronize the
// buffer resources between CPU and GPU.
//
// See also:
// https://developer.apple.com/documentation/metal/synchronizing_a_managed_resource
// https://developer.apple.com/documentation/metal/setting_resource_storage_modes/choosing_a_resource_storage_mode_in_macos
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

template <typename T>
void set_label(T *mtl_obj, const std::string &label) {
  // Set labels on Metal command buffer and encoders, so that they can be
  // tracked in Instrument - Metal System Trace
  if constexpr (std::is_same_v<T, MTLComputeCommandEncoder> ||
                std::is_same_v<T, MTLBlitCommandEncoder> ||
                std::is_same_v<T, MTLCommandBuffer>) {
    auto label_str = mac::wrap_string_as_ns_string(label);
    mac::call(mtl_obj, "setLabel:", label_str.get());
  } else {
    static_assert(always_false_v<T>);
  }
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

void did_modify_range(MTLBuffer *buffer, size_t location, size_t length);

void fill_buffer(MTLBlitCommandEncoder *encoder,
                 MTLBuffer *buffer,
                 mac::TI_NSRange range,
                 uint8_t value);

void copy_from_buffer_to_buffer(MTLBlitCommandEncoder *encoder,
                                MTLBuffer *source_buffer,
                                size_t source_offset,
                                MTLBuffer *destination_buffer,
                                size_t destination_offset,
                                size_t size);

size_t get_max_total_threads_per_threadgroup(
    MTLComputePipelineState *pipeline_state);
#endif  // TI_PLATFORM_OSX

bool is_metal_api_available();

}  // namespace metal
}  // namespace lang
}  // namespace taichi
