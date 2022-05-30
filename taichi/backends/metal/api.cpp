#include "taichi/backends/metal/api.h"

#include "taichi/backends/metal/constants.h"
#include "taichi/util/environ_config.h"

namespace taichi {
namespace lang {
namespace metal {

#ifdef TI_PLATFORM_OSX

extern "C" {
id MTLCreateSystemDefaultDevice();
id MTLCopyAllDevices();
}

namespace {

using mac::call;
using mac::cast_call;
using mac::clscall;
using mac::nsobj_unique_ptr;
using mac::retain_and_wrap_as_nsobj_unique_ptr;
using mac::wrap_as_nsobj_unique_ptr;

}  // namespace

nsobj_unique_ptr<MTLDevice> mtl_create_system_default_device() {
  id dev = MTLCreateSystemDefaultDevice();
  return wrap_as_nsobj_unique_ptr(reinterpret_cast<MTLDevice *>(dev));
}

nsobj_unique_ptr<mac::TI_NSArray> mtl_copy_all_devices() {
  id na = MTLCopyAllDevices();
  return wrap_as_nsobj_unique_ptr(reinterpret_cast<mac::TI_NSArray *>(na));
}

std::string mtl_device_name(MTLDevice *dev) {
  return mac::to_string(cast_call<mac::TI_NSString *>(dev, "name"));
}

nsobj_unique_ptr<MTLCommandQueue> new_command_queue(MTLDevice *dev) {
  auto *queue = cast_call<MTLCommandQueue *>(dev, "newCommandQueue");
  return wrap_as_nsobj_unique_ptr(queue);
}

nsobj_unique_ptr<MTLCommandBuffer> new_command_buffer(MTLCommandQueue *queue) {
  auto *buffer = cast_call<MTLCommandBuffer *>(queue, "commandBuffer");
  return retain_and_wrap_as_nsobj_unique_ptr(buffer);
}

nsobj_unique_ptr<MTLComputeCommandEncoder> new_compute_command_encoder(
    MTLCommandBuffer *buffer) {
  auto *encoder =
      cast_call<MTLComputeCommandEncoder *>(buffer, "computeCommandEncoder");
  return retain_and_wrap_as_nsobj_unique_ptr(encoder);
}

nsobj_unique_ptr<MTLBlitCommandEncoder> new_blit_command_encoder(
    MTLCommandBuffer *buffer) {
  auto *encoder =
      cast_call<MTLBlitCommandEncoder *>(buffer, "blitCommandEncoder");
  return retain_and_wrap_as_nsobj_unique_ptr(encoder);
}

nsobj_unique_ptr<MTLLibrary> new_library_with_source(MTLDevice *device,
                                                     const std::string &source,
                                                     bool fast_math,
                                                     int msl_version) {
  auto source_str = mac::wrap_string_as_ns_string(source);

  id options = clscall("MTLCompileOptions", "alloc");
  options = call(options, "init");
  auto options_cleanup = wrap_as_nsobj_unique_ptr(options);
  call(options, "setFastMathEnabled:", fast_math);
  if (msl_version != kMslVersionNone) {
    call(options, "setLanguageVersion:", msl_version);
  }

  id error_return = nullptr;
  auto *lib = cast_call<MTLLibrary *>(
      device, "newLibraryWithSource:options:error:", source_str.get(), options,
      &error_return);
  if (lib == nullptr) {
    mac::ns_log_object(error_return);
  }
  return wrap_as_nsobj_unique_ptr(lib);
}

nsobj_unique_ptr<MTLFunction> new_function_with_name(MTLLibrary *library,
                                                     const std::string &name) {
  auto name_str = mac::wrap_string_as_ns_string(name);
  auto *func =
      cast_call<MTLFunction *>(library, "newFunctionWithName:", name_str.get());
  return wrap_as_nsobj_unique_ptr(func);
}

nsobj_unique_ptr<MTLComputePipelineState>
new_compute_pipeline_state_with_function(MTLDevice *device,
                                         MTLFunction *function) {
  id error_return = nullptr;
  auto *pipeline_state = cast_call<MTLComputePipelineState *>(
      device, "newComputePipelineStateWithFunction:error:", function,
      &error_return);
  if (pipeline_state == nullptr) {
    mac::ns_log_object(error_return);
  }
  return wrap_as_nsobj_unique_ptr(pipeline_state);
}

nsobj_unique_ptr<MTLBuffer> new_mtl_buffer_no_copy(MTLDevice *device,
                                                   void *ptr,
                                                   size_t length) {
  // MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeManaged
  constexpr int kMtlBufferResourceOptions = 16;

  auto *buffer = cast_call<MTLBuffer *>(
      device, "newBufferWithBytesNoCopy:length:options:deallocator:", ptr,
      length, kMtlBufferResourceOptions, nullptr);
  return wrap_as_nsobj_unique_ptr(buffer);
}

void dispatch_threadgroups(MTLComputeCommandEncoder *encoder,
                           int32_t blocks_x,
                           int32_t blocks_y,
                           int32_t blocks_z,
                           int32_t threads_x,
                           int32_t threads_y,
                           int32_t threads_z) {
  struct MTLSize {
    uint64_t width;
    uint64_t height;
    uint64_t depth;
  };

  MTLSize threadgroups_per_grid;
  threadgroups_per_grid.width = blocks_x;
  threadgroups_per_grid.height = blocks_y;
  threadgroups_per_grid.depth = blocks_z;

  MTLSize threads_per_threadgroup;
  threads_per_threadgroup.width = threads_x;
  threads_per_threadgroup.height = threads_y;
  threads_per_threadgroup.depth = threads_z;

  call(encoder,
       "dispatchThreadgroups:threadsPerThreadgroup:", threadgroups_per_grid,
       threads_per_threadgroup);
}

size_t get_max_total_threads_per_threadgroup(
    MTLComputePipelineState *pipeline_state) {
  // The value of the pointer returned by call is the actual result
  return (size_t)call(pipeline_state, "maxTotalThreadsPerThreadgroup");
}

void did_modify_range(MTLBuffer *buffer, size_t location, size_t length) {
  mac::TI_NSRange range;
  range.location = location;
  range.length = length;
  call(buffer, "didModifyRange:", range);
}

void fill_buffer(MTLBlitCommandEncoder *encoder,
                 MTLBuffer *buffer,
                 mac::TI_NSRange range,
                 uint8_t value) {
  call(encoder, "fillBuffer:bufferrange:rangevalue:", buffer, range, value);
}

void copy_from_buffer_to_buffer(MTLBlitCommandEncoder *encoder,
                                MTLBuffer *source_buffer,
                                size_t source_offset,
                                MTLBuffer *destination_buffer,
                                size_t destination_offset,
                                size_t size) {
  call(encoder, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
       source_buffer, source_offset, destination_buffer, destination_offset,
       size);
}

#endif  // TI_PLATFORM_OSX

bool is_metal_api_available() {
#ifdef TI_PLATFORM_OSX
  if (get_environ_config("TI_ENABLE_METAL", 1) == 0)
    return false;
  // If the macOS is provided by a VM (e.g. Travis CI), it's possible that there
  // is no GPU device, so we still have to do a runtime check.
  auto device = mtl_create_system_default_device();
  return device != nullptr;
#else
  return false;
#endif
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
