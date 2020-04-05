#include "cuda_driver.h"

#include "taichi/system/dynamic_loader.h"

TLANG_NAMESPACE_BEGIN

std::unique_ptr<CUDADriver> CUDADriver::instance;

CUDADriver::CUDADriver() {
#if defined(TI_PLATFORM_LINUX)
  loader = std::make_unique<DynamicLoader>("libcuda.so");
#else
  loader = std::make_unique<DynamicLoader>("nvcuda.dll");
#endif
  loader->load_function("cuMemcpyHtoD_v2", memcpy_host_to_device);
  loader->load_function("cuMemcpyDtoH_v2", memcpy_device_to_host);
  TI_INFO("CUDA driver loaded");
}

CUDADriver &CUDADriver::get_instance() {
  if (!instance)
    instance = std::make_unique<CUDADriver>();
  return *instance;
}

TLANG_NAMESPACE_END
