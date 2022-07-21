#pragma once
#include <taichi/taichi_platform.h>
#include "taichi/taichi_core.h"

namespace capi {
namespace utils {

TI_DLL_EXPORT bool TI_API_CALL is_vulkan_available();
TI_DLL_EXPORT bool TI_API_CALL is_cuda_available();
TI_DLL_EXPORT void TI_API_CALL check_runtime_error(TiRuntime runtime);

}  // namespace utils
}  // namespace capi
