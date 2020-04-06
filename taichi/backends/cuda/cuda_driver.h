#include "cuda.h"

#include "taichi/system/dynamic_loader.h"

TLANG_NAMESPACE_BEGIN

static_assert(sizeof(CUresult) == sizeof(uint32));
static_assert(sizeof(CUmem_advise) == sizeof(uint32));
static_assert(sizeof(CUdevice) == sizeof(uint32));
static_assert(sizeof(CUdevice_attribute) == sizeof(uint32));

std::string get_cuda_error_message(uint32 err);

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

  uint32 call(Args... args) {
    TI_ASSERT(function != nullptr);
    return (uint32)function(args...);
  }

  void call_with_warning(Args... args) {
    auto err = call(args...);
    TI_WARN_IF(err, get_cuda_error_message(err));
  }

  // Note: CUDA driver API passes everything as value
  void operator()(Args... args) {
    auto err = call(args...);
    TI_ERROR_IF(err, get_cuda_error_message(err));
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

  CUDADriverFunction<void *, std::size_t> malloc;
  CUDADriverFunction<void *, std::size_t, uint32> malloc_managed;
  CUDADriverFunction<void *, uint8, std::size_t> memset;
  CUDADriverFunction<void *> mem_free;
  CUDADriverFunction<void *, std::size_t, uint32, uint32> mem_advise;

  CUDADriverFunction<int *, uint32, uint32> device_get_attribute;
  CUDADriverFunction<int> init;
  CUDADriverFunction<int *> device_get_count;
  CUDADriverFunction<int *, void *> device_get;
  CUDADriverFunction<char *, int, void *> device_get_name;
  CUDADriverFunction<char *, int, void *> context_create;
  CUDADriverFunction<void *> context_set_current;
  CUDADriverFunction<void **> context_get_current;
  CUDADriverFunction<std::size_t *, std::size_t *> mem_get_info;

  void (*get_error_name)(uint32, const char **);
  void (*get_error_string)(uint32, const char **);

  CUDADriver();

  ~CUDADriver() = default;

  static CUDADriver &get_instance();
};

TLANG_NAMESPACE_END
