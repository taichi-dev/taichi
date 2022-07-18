#include "c_api_test_utils.h"
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

}  // namespace utils
}  // namespace capi
