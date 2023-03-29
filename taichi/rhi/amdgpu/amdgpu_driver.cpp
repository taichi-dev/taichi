#include "taichi/rhi/amdgpu/amdgpu_driver.h"

#include "taichi/common/dynamic_loader.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#include "taichi/util/environ_config.h"

namespace taichi {
namespace lang {

std::string get_amdgpu_error_message(uint32 err) {
  auto err_name_ptr =
      AMDGPUDriver::get_instance_without_context().get_error_name(err);
  auto err_string_ptr =
      AMDGPUDriver::get_instance_without_context().get_error_string(err);
  return fmt::format("AMDGPU Error {}: {}", err_name_ptr, err_string_ptr);
}

AMDGPUDriverBase::AMDGPUDriverBase() {
  disabled_by_env_ = (get_environ_config("TI_ENABLE_AMDGPU", 1) == 0);
  if (disabled_by_env_) {
    TI_TRACE(
        "AMDGPU driver disabled by enviroment variable \"TI_ENABLE_AMDGPU\".");
  }
}

bool AMDGPUDriverBase::load_lib(std::string lib_linux) {
#if defined(TI_PLATFORM_LINUX)
  auto lib_name = lib_linux;
#else
  static_assert(false, "Taichi AMDGPU driver supports only Linux.");
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

bool AMDGPUDriver::detected() {
  return !disabled_by_env_ && loader_->loaded();
}

AMDGPUDriver::AMDGPUDriver() {
  if (!load_lib("libamdhip64.so"))
    return;

  loader_->load_function("hipGetErrorName", get_error_name);
  loader_->load_function("hipGetErrorString", get_error_string);
  loader_->load_function("hipDriverGetVersion", driver_get_version);

  int version;
  driver_get_version(&version);
  TI_TRACE("AMDGPU driver API (v{}.{}) loaded.", version / 1000,
           version % 1000 / 10);

#define PER_AMDGPU_FUNCTION(name, symbol_name, ...) \
  name.set(loader_->load_function(#symbol_name));   \
  name.set_lock(&lock_);                            \
  name.set_names(#name, #symbol_name);
#include "taichi/rhi/amdgpu/amdgpu_driver_functions.inc.h"
#undef PER_AMDGPU_FUNCTION
}

AMDGPUDriver &AMDGPUDriver::get_instance_without_context() {
  // Thread safety guaranteed by C++ compiler
  // Note this is never deleted until the process finishes
  static AMDGPUDriver *instance = new AMDGPUDriver();
  return *instance;
}

AMDGPUDriver &AMDGPUDriver::get_instance() {
  // initialize the AMDGPU context so that the driver APIs can be called later
  AMDGPUContext::get_instance();
  return get_instance_without_context();
}

}  // namespace lang
}  // namespace taichi
