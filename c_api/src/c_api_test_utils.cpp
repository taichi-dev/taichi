#include "c_api_test_utils.h"
#include "taichi_llvm_impl.h"
#include "taichi/platform/cuda/detect_cuda.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif

namespace capi {
namespace utils {

bool is_vulkan_available() {
#ifdef TI_WITH_VULKAN
  return taichi::lang::vulkan::is_vulkan_api_available();
#else
  return false;
#endif
}

bool is_cuda_available() {
  return taichi::is_cuda_api_available();
}

void check_runtime_error(TiRuntime runtime) {
#ifdef TI_WITH_LLVM
  auto *llvm_runtime = dynamic_cast<capi::LlvmRuntime *>((Runtime *)runtime);
  if (!llvm_runtime) {
    TI_NOT_IMPLEMENTED;
  }
  llvm_runtime->check_runtime_error();
#else
  TI_NOT_IMPLEMENTED;
#endif
}

}  // namespace utils
}  // namespace capi
