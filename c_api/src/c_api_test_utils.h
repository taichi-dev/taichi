#pragma once
#include <taichi/taichi_platform.h>

namespace capi {
namespace utils {

TI_DLL_EXPORT bool TI_API_CALL is_vulkan_available();
TI_DLL_EXPORT bool TI_API_CALL is_cuda_available();

}  // namespace utils
}  // namespace capi
