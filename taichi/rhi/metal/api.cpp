#include "taichi/rhi/metal/api.h"

#include "taichi/rhi/metal/constants.h"
#include "taichi/util/environ_config.h"

namespace taichi::lang {
namespace metal {

#ifdef TI_PLATFORM_OSX

namespace {

using mac::call;
using mac::cast_call;
using mac::clscall;
using mac::nsobj_unique_ptr;
using mac::retain_and_wrap_as_nsobj_unique_ptr;
using mac::wrap_as_nsobj_unique_ptr;

}  // namespace

nsobj_unique_ptr<MTL::Device> mtl_create_system_default_device() {
  return wrap_as_nsobj_unique_ptr(MTL::CreateSystemDefaultDevice());
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
}  // namespace taichi::lang
