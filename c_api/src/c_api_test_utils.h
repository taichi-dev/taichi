#pragma once
#include "taichi_core_impl.h"

namespace capi {
namespace utils {

TI_DLL_EXPORT void TI_API_CALL check_runtime_error(TiRuntime runtime);

TI_DLL_EXPORT bool TI_API_CALL check_cuda_value(void *ptr, float value);
TI_DLL_EXPORT bool TI_API_CALL check_cuda_value(void *ptr, double value);

TI_DLL_EXPORT uint16_t to_float16(float in);
TI_DLL_EXPORT float to_float32(uint16_t in);

TI_DLL_EXPORT void TI_API_CALL cuda_malloc(void **ptr, size_t size);
TI_DLL_EXPORT void TI_API_CALL cuda_memcpy_host_to_device(void *ptr,
                                                          void *data,
                                                          size_t size);
TI_DLL_EXPORT void TI_API_CALL cuda_memcpy_device_to_host(void *ptr,
                                                          void *data,
                                                          size_t size);
}  // namespace utils
}  // namespace capi
