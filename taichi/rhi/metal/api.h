#pragma once

// Reference implementation:
// https://github.com/halide/Halide/blob/master/src/runtime/metal.cpp

#include <string>
#include <Metal/Metal.hpp>

#include "taichi/common/trait.h"
#include "taichi/util/lang_util.h"
#include "taichi/platform/mac/objc_api.h"
#include "taichi/rhi/metal/metal_api.h"

namespace taichi::lang {
namespace metal {

// Expose these incomplete structs so that other modules (e.g. MetalRuntime)
// don't have to be compiled conditionally.
typedef MTL::Device MTLDevice;
typedef MTL::Library MTLLibrary;
typedef MTL::ComputePipelineState MTLComputePipelineState;
typedef MTL::CommandQueue MTLCommandQueue;
typedef MTL::CommandBuffer MTLCommandBuffer;
typedef MTL::ComputeCommandEncoder MTLComputeCommandEncoder;
typedef MTL::BlitCommandEncoder MTLBlitCommandEncoder;
typedef MTL::Function MTLFunction;
typedef MTL::ComputePipelineState MTLComputePipelineState;
typedef MTL::Buffer MTLBuffer;

#ifdef TI_PLATFORM_OSX

using mac::nsobj_unique_ptr;

nsobj_unique_ptr<MTLDevice> mtl_create_system_default_device();

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

#endif  // TI_PLATFORM_OSX

bool is_metal_api_available();

}  // namespace metal
}  // namespace taichi::lang
