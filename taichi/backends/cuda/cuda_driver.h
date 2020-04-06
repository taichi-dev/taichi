#include "taichi/system/dynamic_loader.h"
#include "taichi/backends/cuda/cuda_utils.h"

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
    auto ret = function(args...);
    // check_cuda_error(ret);
  }
};

class CUDADriver {
 private:
  std::unique_ptr<DynamicLoader> loader;

  static std::unique_ptr<CUDADriver> instance;

 public:
  CUDADriverFunction<void *, void *, std::size_t> memcpy_host_to_device;
  CUDADriverFunction<void *, void *, std::size_t> memcpy_device_to_host;
  CUDADriverFunction<void *> memfree;

  CUDADriver();

  ~CUDADriver() = default;

  static CUDADriver &get_instance();
};

TLANG_NAMESPACE_END
