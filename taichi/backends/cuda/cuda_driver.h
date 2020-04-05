#include "taichi/system/dynamic_loader.h"

TLANG_NAMESPACE_BEGIN

class CUDADriver {
 private:
  std::unique_ptr<DynamicLoader> loader;

  static std::unique_ptr<CUDADriver> instance;

 public:
  void (*memcpy_host_to_device)(void *, void *, std::size_t);
  void (*memcpy_device_to_host)(void *, void *, std::size_t);

  CUDADriver();

  ~CUDADriver() = default;
  // memcpy = lo

  static CUDADriver &get_instance();
};

TLANG_NAMESPACE_END
