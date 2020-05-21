#pragma once

#include "taichi/system/dynamic_loader.h"

#if (0)
// Turn on to check for comptability
namespace taichi {
static_assert(sizeof(CUresult) == sizeof(uint32));
static_assert(sizeof(CUmem_advise) == sizeof(uint32));
static_assert(sizeof(CUdevice) == sizeof(uint32));
static_assert(sizeof(CUdevice_attribute) == sizeof(uint32));
static_assert(sizeof(CUfunction) == sizeof(void *));
static_assert(sizeof(CUmodule) == sizeof(void *));
static_assert(sizeof(CUstream) == sizeof(void *));
static_assert(sizeof(CUevent) == sizeof(void *));
static_assert(sizeof(CUjit_option) == sizeof(uint32));
}  // namespace taichi
#endif

TLANG_NAMESPACE_BEGIN

// Driver constants from cuda.h

constexpr uint32 CU_EVENT_DEFAULT = 0x0;
constexpr uint32 CU_MEM_ATTACH_GLOBAL = 0x1;
constexpr uint32 CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3;
constexpr uint32 CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2;
constexpr uint32 CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
constexpr uint32 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr uint32 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

std::string get_cuda_error_message(uint32 err);

template <typename... Args>
class CUDADriverFunction {
 public:
  CUDADriverFunction() {
    function = nullptr;
  }

  void set(void *func_ptr) {
    function = (func_type *)func_ptr;
  }

  uint32 call(Args... args) {
    TI_ASSERT(function != nullptr);
    return (uint32)function(args...);
  }

  void set_names(const std::string &name, const std::string &symbol_name) {
    this->name = name;
    this->symbol_name = symbol_name;
  }

  std::string get_error_message(uint32 err) {
    return get_cuda_error_message(err) +
           fmt::format(" while calling {} ({})", name, symbol_name);
  }

  void call_with_warning(Args... args) {
    auto err = call(args...);
    TI_WARN_IF(err, get_error_message(err));
  }

  // Note: CUDA driver API passes everything as value
  void operator()(Args... args) {
    auto err = call(args...);
    TI_ERROR_IF(err, get_error_message(err));
  }

 private:
  using func_type = uint32_t(Args...);
  func_type *function;
  std::string name, symbol_name;
};

class CUDADriver {
 public:
#define PER_CUDA_FUNCTION(name, symbol_name, ...) \
  CUDADriverFunction<__VA_ARGS__> name;
#include "taichi/backends/cuda/cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION

  void (*get_error_name)(uint32, const char **);

  void (*get_error_string)(uint32, const char **);

  bool detected();

  ~CUDADriver() = default;

  static CUDADriver &get_instance();

  static CUDADriver &get_instance_without_context();

 private:
  CUDADriver();

  std::unique_ptr<DynamicLoader> loader;
};

TLANG_NAMESPACE_END
