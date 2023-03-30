#pragma once
#include "taichi_core_impl.h"

namespace capi {
namespace utils {

TI_DLL_EXPORT void TI_API_CALL check_runtime_error(TiRuntime runtime);

TI_DLL_EXPORT bool TI_API_CALL check_cuda_value(void *ptr, float value);
TI_DLL_EXPORT bool TI_API_CALL check_cuda_value(void *ptr, double value);

TI_DLL_EXPORT uint16_t to_float16(float in);
TI_DLL_EXPORT float to_float32(uint16_t in);

}  // namespace utils
}  // namespace capi
