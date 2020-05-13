#include "taichi/backends/cuda/cuda_driver.h"

#include "taichi/system/dynamic_loader.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/util/environ_config.h"

TLANG_NAMESPACE_BEGIN

std::string get_cuda_error_message(uint32 err) {
  const char *err_name_ptr;
  const char *err_string_ptr;
  CUDADriver::get_instance_without_context().get_error_name(err, &err_name_ptr);
  CUDADriver::get_instance_without_context().get_error_string(err,
                                                              &err_string_ptr);
  return fmt::format("CUDA Error {}: {}", err_name_ptr, err_string_ptr);
}

bool CUDADriver::detected() {
  if (get_environ_config("TI_ENABLE_CUDA", 1) == 0)
    return false;
  return loader->loaded();
}

CUDADriver::CUDADriver() {
#if defined(TI_PLATFORM_LINUX)
  loader = std::make_unique<DynamicLoader>("libcuda.so");
#elif defined(TI_PLATFORM_WINDOWS)
  loader = std::make_unique<DynamicLoader>("nvcuda.dll");
#else
  static_assert(false, "Taichi CUDA driver supports only Windows and Linux.");
#endif

  if (detected()) {
    loader->load_function("cuGetErrorName", get_error_name);
    loader->load_function("cuGetErrorString", get_error_string);

#define PER_CUDA_FUNCTION(name, symbol_name, ...) \
  name.set(loader->load_function(#symbol_name));  \
  name.set_names(#name, #symbol_name);
#include "taichi/backends/cuda/cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION

    int version;
    driver_get_version(&version);

    TI_TRACE("CUDA driver API (v{}.{}) loaded.", version / 1000,
             version % 1000 / 10);
  } else {
    TI_DEBUG("CUDA driver not found.");
  }
}

// This is for initializing the CUDA context itself
CUDADriver &CUDADriver::get_instance_without_context() {
  // Thread safety guaranteed by C++ compiler
  // Note this is never deleted until the process finishes
  static CUDADriver *instance = new CUDADriver();
  return *instance;
}

CUDADriver &CUDADriver::get_instance() {
  // initialize the CUDA context so that the driver APIs can be called later
  CUDAContext::get_instance();
  return get_instance_without_context();
}

TLANG_NAMESPACE_END
