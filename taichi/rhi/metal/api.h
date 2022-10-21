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

#endif  // TI_PLATFORM_OSX

bool is_metal_api_available();

}  // namespace metal
}  // namespace taichi::lang
