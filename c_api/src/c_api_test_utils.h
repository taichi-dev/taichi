#pragma once
#include "taichi_core_impl.h"

namespace capi {
namespace utils {

TI_DLL_EXPORT bool TI_API_CALL is_vulkan_available();
TI_DLL_EXPORT bool TI_API_CALL is_opengl_available();
TI_DLL_EXPORT bool TI_API_CALL is_cuda_available();
TI_DLL_EXPORT void TI_API_CALL check_runtime_error(TiRuntime runtime);

TI_DLL_EXPORT bool TI_API_CALL check_cuda_value(void *ptr, float value);
TI_DLL_EXPORT bool TI_API_CALL check_cuda_value(void *ptr, double value);

}  // namespace utils
}  // namespace capi
