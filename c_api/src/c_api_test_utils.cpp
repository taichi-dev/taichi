#include "c_api_test_utils.h"
#include "taichi_llvm_impl.h"

#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_driver.h"
#endif

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif

#ifdef TI_WITH_OPENGL
#include "taichi/rhi/opengl/opengl_api.h"
#endif

namespace capi {
namespace utils {

template <typename T>
bool check_cuda_value_impl(void *ptr, T value) {
#ifdef TI_WITH_CUDA
  T host_val;
  taichi::lang::CUDADriver::get_instance().memcpy_device_to_host(&host_val, ptr,
                                                                 sizeof(T));
  if (host_val == value)
    return true;
#endif
  return false;
}

bool check_cuda_value(void *ptr, float value) {
  return check_cuda_value_impl(ptr, value);
}

bool check_cuda_value(void *ptr, double value) {
  return check_cuda_value_impl(ptr, value);
}

void check_runtime_error(TiRuntime runtime) {
#ifdef TI_WITH_LLVM
  auto *llvm_runtime = dynamic_cast<capi::LlvmRuntime *>((Runtime *)runtime);
  if (!llvm_runtime) {
    ti_set_last_error(TI_ERROR_INVALID_STATE, "llvm_runtime");
  }
  llvm_runtime->check_runtime_error();
#else
  ti_set_last_error(TI_ERROR_INVALID_STATE, "llvm_runtime");
#endif
}

}  // namespace utils
}  // namespace capi
