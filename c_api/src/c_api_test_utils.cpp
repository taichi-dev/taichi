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

void cuda_malloc(void **ptr, size_t size) {
#ifdef TI_WITH_CUDA
  taichi::lang::CUDADriver::get_instance().malloc(ptr, size);
#endif
}

void cuda_memcpy_host_to_device(void *ptr, void *data, size_t size) {
#ifdef TI_WITH_CUDA
  taichi::lang::CUDADriver::get_instance().memcpy_host_to_device(ptr, data,
                                                                 size);
#endif
}

void cuda_memcpy_device_to_host(void *ptr, void *data, size_t size) {
#ifdef TI_WITH_CUDA
  taichi::lang::CUDADriver::get_instance().memcpy_device_to_host(ptr, data,
                                                                 size);
#endif
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

static void float32(float *__restrict out, const uint16_t in) {
  uint32_t t1;
  uint32_t t2;
  uint32_t t3;

  t1 = in & 0x7fffu;  // Non-sign bits
  t2 = in & 0x8000u;  // Sign bit
  t3 = in & 0x7c00u;  // Exponent

  t1 <<= 13u;  // Align mantissa on MSB
  t2 <<= 16u;  // Shift sign bit into position

  t1 += 0x38000000;  // Adjust bias

  t1 = (t3 == 0 ? 0 : t1);  // Denormals-as-zero

  t1 |= t2;  // Re-insert sign bit

  *((uint32_t *)out) = t1;
};

static void float16(uint16_t *__restrict out, const float in) {
  uint32_t inu = *((uint32_t *)&in);
  uint32_t t1;
  uint32_t t2;
  uint32_t t3;

  t1 = inu & 0x7fffffffu;  // Non-sign bits
  t2 = inu & 0x80000000u;  // Sign bit
  t3 = inu & 0x7f800000u;  // Exponent

  t1 >>= 13u;  // Align mantissa on MSB
  t2 >>= 16u;  // Shift sign bit into position

  t1 -= 0x1c000;  // Adjust bias

  t1 = (t3 < 0x38800000u) ? 0 : t1;       // Flush-to-zero
  t1 = (t3 > 0x8e000000u) ? 0x7bff : t1;  // Clamp-to-max
  t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero

  t1 |= t2;  // Re-insert sign bit

  *((uint16_t *)out) = t1;
};

uint16_t to_float16(float in) {
  uint16_t out;
  float16(&out, in);
  return out;
}

float to_float32(uint16_t in) {
  float out;
  float32(&out, in);
  return out;
}

}  // namespace utils
}  // namespace capi
