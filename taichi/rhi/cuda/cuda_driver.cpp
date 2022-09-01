#include "taichi/rhi/cuda/cuda_driver.h"

#include "taichi/system/dynamic_loader.h"
#include "taichi/rhi/cuda/cuda_context.h"
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

CUDADriverBase::CUDADriverBase() {
  disabled_by_env_ = (get_environ_config("TI_ENABLE_CUDA", 1) == 0);
  if (disabled_by_env_) {
    TI_TRACE("CUDA driver disabled by enviroment variable \"TI_ENABLE_CUDA\".");
  }
}

bool CUDADriverBase::load_lib(std::string lib_linux, std::string lib_windows) {
#if defined(TI_PLATFORM_LINUX)
  auto lib_name = lib_linux;
#elif defined(TI_PLATFORM_WINDOWS)
  auto lib_name = lib_windows;
#else
  static_assert(false, "Taichi CUDA driver supports only Windows and Linux.");
#endif

  loader_ = std::make_unique<DynamicLoader>(lib_name);
  if (!loader_->loaded()) {
    TI_WARN("{} lib not found.", lib_name);
    return false;
  } else {
    TI_TRACE("{} loaded!", lib_name);
    return true;
  }
}

bool CUDADriver::detected() {
  return !disabled_by_env_ && cuda_version_valid_ && loader_->loaded();
}

CUDADriver::CUDADriver() {
  if (!load_lib("libcuda.so", "nvcuda.dll"))
    return;

  loader_->load_function("cuGetErrorName", get_error_name);
  loader_->load_function("cuGetErrorString", get_error_string);
  loader_->load_function("cuDriverGetVersion", driver_get_version);

  int version;
  driver_get_version(&version);
  TI_TRACE("CUDA driver API (v{}.{}) loaded.", version / 1000,
           version % 1000 / 10);

  // CUDA versions should >= 10.
  if (version < 10000) {
    TI_WARN("The Taichi CUDA backend requires at least CUDA 10.0, got v{}.{}.",
            version / 1000, version % 1000 / 10);
    return;
  }

  cuda_version_valid_ = true;
#define PER_CUDA_FUNCTION(name, symbol_name, ...) \
  name.set(loader_->load_function(#symbol_name)); \
  name.set_lock(&lock_);                          \
  name.set_names(#name, #symbol_name);
#include "taichi/rhi/cuda/cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION
}

// This is for initializing the CUDA driver itself
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

CUSPARSEDriver::CUSPARSEDriver() {
}

CUSPARSEDriver &CUSPARSEDriver::get_instance() {
  static CUSPARSEDriver *instance = new CUSPARSEDriver();
  return *instance;
}

bool CUSPARSEDriver::load_cusparse() {
  cusparse_loaded_ = load_lib("libcusparse.so", "cusparse64_11.dll");

  if (!cusparse_loaded_) {
    return false;
  }
#define PER_CUSPARSE_FUNCTION(name, symbol_name, ...) \
  name.set(loader_->load_function(#symbol_name));     \
  name.set_lock(&lock_);                              \
  name.set_names(#name, #symbol_name);
#include "taichi/rhi/cuda/cusparse_functions.inc.h"
#undef PER_CUSPARSE_FUNCTION
  return cusparse_loaded_;
}

CUSOLVERDriver::CUSOLVERDriver() {
}

CUSOLVERDriver &CUSOLVERDriver::get_instance() {
  static CUSOLVERDriver *instance = new CUSOLVERDriver();
  return *instance;
}

bool CUSOLVERDriver::load_cusolver() {
  cusolver_loaded_ = load_lib("libcusolver.so", "cusolver64_11.dll");
  if (!cusolver_loaded_) {
    return false;
  }
#define PER_CUSOLVER_FUNCTION(name, symbol_name, ...) \
  name.set(loader_->load_function(#symbol_name));     \
  name.set_lock(&lock_);                              \
  name.set_names(#name, #symbol_name);
#include "taichi/rhi/cuda/cusolver_functions.inc.h"
#undef PER_CUSOLVER_FUNCTION
  return cusolver_loaded_;
}
TLANG_NAMESPACE_END
