#pragma once

#include "cuda.h"

#include "taichi/system/dynamic_loader.h"

TLANG_NAMESPACE_BEGIN

static_assert(sizeof(CUresult) == sizeof(uint32));
static_assert(sizeof(CUmem_advise) == sizeof(uint32));
static_assert(sizeof(CUdevice) == sizeof(uint32));
static_assert(sizeof(CUdevice_attribute) == sizeof(uint32));
static_assert(sizeof(CUfunction) == sizeof(void *));
static_assert(sizeof(CUmodule) == sizeof(void *));
static_assert(sizeof(CUstream) == sizeof(void *));
static_assert(sizeof(CUevent) == sizeof(void *));
static_assert(sizeof(CUjit_option) == sizeof(uint32));

std::string get_cuda_error_message(uint32 err);

template <typename... Args>
class CUDADriverFunction {
 private:
  using func_type = uint32_t(Args...);
  func_type *function;
  std::string name, symbol_name;

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
};

class CUDADriver {
 private:
  std::unique_ptr<DynamicLoader> loader;

  static std::unique_ptr<CUDADriver> instance;

 public:
#define PER_CUDA_FUNCTION(name, symbol_name, ...) \
  CUDADriverFunction<__VA_ARGS__> name;
#include "taichi/backends/cuda/cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION

  void (*get_error_name)(uint32, const char **);
  void (*get_error_string)(uint32, const char **);

  CUDADriver();

  ~CUDADriver() = default;

  static CUDADriver &get_instance();

  static CUDADriver &get_instance_without_context();
};

TLANG_NAMESPACE_END
