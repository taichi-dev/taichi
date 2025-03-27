#pragma once

#include <mutex>

#include "taichi/common/dynamic_loader.h"

namespace taichi {
namespace lang {

constexpr uint32 HIP_EVENT_DEFAULT = 0x0;
constexpr uint32 HIP_STREAM_DEFAULT = 0x0;
constexpr uint32 HIP_STREAM_NON_BLOCKING = 0x1;
constexpr uint32 HIP_MEM_ATTACH_GLOBAL = 0x1;
constexpr uint32 HIP_MEM_ADVISE_SET_PREFERRED_LOCATION = 3;
constexpr uint32 HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 26;
constexpr uint32 HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 63;
// sizeof(hipDeviceProperties_t) in ROCm 6.
// ROCm 5.7.1 is 792 and ROCm 6 is 1472, so to make both work we use whichever
// is larger.
constexpr uint32 HIP_DEVICE_PROPERTIES_STRUCT_SIZE = 1472;
// offsetof(hipDeviceProp_t, gcnArchName) / 4
constexpr uint32 HIP_DEVICE_GCN_ARCH_NAME = 396 / 4;
// offsetof(hipDeviceProp_t, gcnArchName) / 4
constexpr uint32 HIP_DEVICE_GCN_ARCH_NAME_6 = 1160 / 4;
// offsetof(hipDeviceProp_t, major) / 4
constexpr uint32 HIP_DEVICE_MAJOR = 328 / 4;
// offsetof(hipDeviceProp_t, major) / 4
constexpr uint32 HIP_DEVICE_MAJOR_6 = 360 / 4;
// offsetof(hipDeviceProp_t, minor) / 4
constexpr uint32 HIP_DEVICE_MINOR = 332 / 4;
// offsetof(hipDeviceProp_t, minor) / 4
constexpr uint32 HIP_DEVICE_MINOR_6 = 364 / 4;
constexpr uint32 HIP_ERROR_ASSERT = 710;
constexpr uint32 HIP_JIT_MAX_REGISTERS = 0;
constexpr uint32 HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2;
constexpr uint32 HIP_SUCCESS = 0;
constexpr uint32 HIP_MEMORYTYPE_DEVICE = 1;

std::string get_amdgpu_error_message(uint32 err);

template <typename... Args>
class AMDGPUFunction {
 public:
  AMDGPUFunction() {
    function_ = nullptr;
  }

  void set(void *func_ptr) {
    function_ = (func_type *)func_ptr;
  }

  uint32 call(Args... args) {
    TI_ASSERT(function_ != nullptr);
    TI_ASSERT(driver_lock_ != nullptr);
    std::lock_guard<std::mutex> _(*driver_lock_);
    return (uint32)function_(args...);
  }

  void set_names(const std::string &name, const std::string &symbol_name) {
    name_ = name;
    symbol_name_ = symbol_name;
  }

  void set_lock(std::mutex *lock) {
    driver_lock_ = lock;
  }

  std::string get_error_message(uint32 err) {
    return get_amdgpu_error_message(err) +
           fmt::format(" while calling {} ({})", name_, symbol_name_);
  }

  uint32 call_with_warning(Args... args) {
    auto err = call(args...);
    TI_WARN_IF(err, "{}", get_error_message(err));
    return err;
  }

  void operator()(Args... args) {
    auto err = call(args...);
    TI_ERROR_IF(err, get_error_message(err));
  }

 private:
  using func_type = uint32_t(Args...);

  func_type *function_{nullptr};
  std::string name_, symbol_name_;
  std::mutex *driver_lock_{nullptr};
};

class AMDGPUDriverBase {
 public:
  ~AMDGPUDriverBase() = default;

 protected:
  std::unique_ptr<DynamicLoader> loader_;
  AMDGPUDriverBase();

  bool load_lib(std::string lib_linux);

  bool disabled_by_env_{false};
};

class AMDGPUDriver : protected AMDGPUDriverBase {
 public:
#define PER_AMDGPU_FUNCTION(name, symbol_name, ...) \
  AMDGPUFunction<__VA_ARGS__> name;
#include "taichi/rhi/amdgpu/amdgpu_driver_functions.inc.h"
#undef PER_AMDGPU_FUNCTION

  char *(*get_error_name)(uint32);

  char *(*get_error_string)(uint32);

  void (*driver_get_version)(int *);

  void (*runtime_get_version)(int *);

  bool detected();

  static AMDGPUDriver &get_instance();

  static AMDGPUDriver &get_instance_without_context();

 private:
  AMDGPUDriver();

  std::mutex lock_;

  // bool rocm_version_valid_{false};
};

}  // namespace lang
}  // namespace taichi
