#include "cuda_driver.h"

#include "taichi/system/dynamic_loader.h"
#include "taichi/backends/cuda/cuda_context.h"

TLANG_NAMESPACE_BEGIN

std::string get_cuda_error_message(uint32 err) {
  const char *err_name_ptr;
  const char *err_string_ptr;
  CUDADriver::get_instance().get_error_name(err, &err_name_ptr);
  CUDADriver::get_instance().get_error_string(err, &err_string_ptr);
  return fmt::format("CUDA Error {}: {}", err_name_ptr, err_string_ptr);
}

std::unique_ptr<CUDADriver> CUDADriver::instance;

CUDADriver::CUDADriver() {
#if defined(TI_PLATFORM_LINUX)
  loader = std::make_unique<DynamicLoader>("libcuda.so");
#else
  loader = std::make_unique<DynamicLoader>("nvcuda.dll");
#endif

  loader->load_function("cuGetErrorName", get_error_name);
  loader->load_function("cuGetErrorString", get_error_string);

  memcpy_host_to_device.set(loader->load_function("cuMemcpyHtoD_v2"));
  memcpy_device_to_host.set(loader->load_function("cuMemcpyDtoH_v2"));
  malloc.set(loader->load_function("cuMemAlloc_v2"));
  malloc_managed.set(loader->load_function("cuMemAllocManaged"));
  memset.set(loader->load_function("cuMemsetD8_v2"));
  mem_advise.set(loader->load_function("cuMemAdvise"));
  mem_free.set(loader->load_function("cuMemFree_v2"));
  device_get_attribute.set(loader->load_function("cuDeviceGetAttribute"));

  TI_INFO("CUDA driver loaded");
}

CUDADriver &CUDADriver::get_instance() {
  CUDAContext::get_instance();
  if (!instance)
    instance = std::make_unique<CUDADriver>();
  return *instance;
}

TLANG_NAMESPACE_END
