#include "taichi/rhi/cuda/cuda_driver.h"

#include "taichi/common/dynamic_loader.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/util/environ_config.h"

namespace taichi::lang {

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

bool CUDADriverBase::check_lib_loaded(std::string lib_linux,
                                      std::string lib_windows) {
#if defined(TI_PLATFORM_LINUX)
  auto lib_name = lib_linux;
#elif defined(TI_PLATFORM_WINDOWS)
  auto lib_name = lib_windows;
#else
  static_assert(false, "Taichi CUDA driver supports only Windows and Linux.");
#endif

  return DynamicLoader::check_lib_loaded(lib_name);
}

std::string get_lib_name_linux(const std::string &lib_name, int version) {
  return "lib" + lib_name + ".so." + std::to_string(version);
}

std::string get_lib_name_windows(const std::string &lib_name,
                                 const std::string &win_arch_name,
                                 int version) {
  return lib_name + win_arch_name + std::to_string(version) + ".dll";
}

bool CUDADriverBase::try_load_lib_any_version(
    const std::string &lib_name,
    const std::string &win_arch_name,
    const std::vector<int> &versions_to_try) {
  // Check if any versions of this lib are already loaded.
  for (auto version : versions_to_try) {
    std::string lib_name_linux = get_lib_name_linux(lib_name, version);
    std::string lib_name_windows =
        get_lib_name_windows(lib_name, win_arch_name, version);
    if (check_lib_loaded(lib_name_linux, lib_name_windows)) {
      load_lib(lib_name_linux, lib_name_windows);
      return true;
    }
  }

  // Try load any version of this lib if none of them are loaded.
  bool loaded = false;
  if (!loaded) {
#ifdef WIN32
    for (auto version : versions_to_try) {
      std::string lib_name_windows =
          get_lib_name_windows(lib_name, win_arch_name, version);
      loader_ = std::make_unique<DynamicLoader>(lib_name_windows);
      loaded = loader_->loaded();
      if (loaded) {
        break;
      }
    }
#else
    for (auto version : versions_to_try) {
      std::string lib_name_linux = get_lib_name_linux(lib_name, version);
      loader_ = std::make_unique<DynamicLoader>(lib_name_linux);
      loaded = loader_->loaded();
      if (loaded) {
        break;
      }
    }
    if (!loaded) {
      // Use the default version on linux.
      std::string lib_name_linux = "lib" + lib_name + ".so";
      loader_ = std::make_unique<DynamicLoader>(lib_name_linux);
      loaded = loader_->loaded();
    }
#endif
  }
  return loaded;
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
  version_major_ = version / 1000;
  version_minor_ = version % 1000 / 10;

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

void CUDADriver::malloc_async(void **dev_ptr, size_t size, CUstream stream) {
  if (CUDAContext::get_instance().supports_mem_pool()) {
    malloc_async_impl(dev_ptr, size, stream);
  } else {
    malloc(dev_ptr, size);
  }
}

void CUDADriver::mem_free_async(void *dev_ptr, CUstream stream) {
  if (CUDAContext::get_instance().supports_mem_pool()) {
    mem_free_async_impl(dev_ptr, stream);
  } else {
    mem_free(dev_ptr);
  }
}

CUSPARSEDriver::CUSPARSEDriver() {
}

CUSPARSEDriver &CUSPARSEDriver::get_instance() {
  static CUSPARSEDriver *instance = new CUSPARSEDriver();
  return *instance;
}

bool CUSPARSEDriver::load_cusparse() {
  /*
  Load the cuSparse lib whose version follows the CUDA driver's version.
  See load_cusolver() for more information.
  */
  // Get the CUDA Driver's version
  int cuda_version = CUDADriver::get_instance().get_version_major();
  // Try to load the cusparse lib whose version is derived from the CUDA driver
  cusparse_loaded_ = try_load_lib_any_version("cusparse", "64_",
                                              {cuda_version, cuda_version - 1});
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
  /*
  Load the cuSolver lib whose version follows the CUDA driver's version.
  Note that cusolver's filename is NOT necessarily the same with CUDA Toolkit
  (on Windows). For instance, CUDA Toolkit 12.2 ships a cusolver64_11.dll
  (checked on 2023.7.13) Therefore, the following function attempts to load a
  cusolver lib which is one version backward from the CUDA Driver's version.
  */
  // Get the CUDA Driver's version
  int cuda_version = CUDADriver::get_instance().get_version_major();
  // Try to load the cusolver lib whose version is derived from the CUDA driver
  cusolver_loaded_ = try_load_lib_any_version("cusolver", "64_",
                                              {cuda_version, cuda_version - 1});
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

CUBLASDriver::CUBLASDriver() {
}

CUBLASDriver &CUBLASDriver::get_instance() {
  static CUBLASDriver *instance = new CUBLASDriver();
  return *instance;
}

bool CUBLASDriver::load_cublas() {
  /* To be compatible with torch environment, please libcublas.so.11 other than
   * libcublas.so. When using libcublas.so, the system cublas will be loaded and
   * it would confict with torch's cublas. When using libcublas.so.11, the
   * torch's cublas will be loaded.
   */
  cublas_loaded_ = try_load_lib_any_version("cublas", "64_", {11, 12, 10});
  if (!cublas_loaded_) {
    return false;
  }
#define PER_CUBLAS_FUNCTION(name, symbol_name, ...) \
  name.set(loader_->load_function(#symbol_name));   \
  name.set_lock(&lock_);                            \
  name.set_names(#name, #symbol_name);
#include "taichi/rhi/cuda/cublas_functions.inc.h"
#undef PER_CUBLAS_FUNCTION
  return cublas_loaded_;
}

}  // namespace taichi::lang
