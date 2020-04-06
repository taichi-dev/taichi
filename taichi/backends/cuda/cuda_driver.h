#include "cuda.h"

#include "taichi/system/dynamic_loader.h"

TLANG_NAMESPACE_BEGIN

template <typename... Args>
class CUDADriverFunction {
 private:
  using func_type = uint32_t(Args...);
  func_type *function;

 public:
  CUDADriverFunction() {
    function = nullptr;
  }

  void set(void *func_ptr) {
    function = (func_type *)func_ptr;
  }

  void operator()(Args... args) {
    TI_ASSERT(function != nullptr);
    // function(args...);
    auto ret = (CUresult)function(args...);
    if (ret) {
      const char *err_name_ptr;
      const char *err_string_ptr;
      cuGetErrorName(ret, &err_name_ptr);
      cuGetErrorString(ret, &err_string_ptr);
      TI_ERROR("CUDA Error {}: {}", err_name_ptr, err_string_ptr);
    }
  }
};

class CUDADriver {
 private:
  std::unique_ptr<DynamicLoader> loader;

  static std::unique_ptr<CUDADriver> instance;

 public:
  CUDADriverFunction<void *, void *, std::size_t> memcpy_host_to_device;
  CUDADriverFunction<void *, void *, std::size_t> memcpy_device_to_host;
  CUDADriverFunction<void *, std::size_t, std::uint32_t> malloc_managed;
  CUDADriverFunction<void *> memfree;

  CUDADriver();

  ~CUDADriver() = default;

  static CUDADriver &get_instance();
};

TLANG_NAMESPACE_END
