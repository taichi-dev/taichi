#include "taichi/platform/cuda/detect_cuda.h"

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_driver.h"
#endif

namespace taichi {

bool is_cuda_api_available() {
#if defined(TI_WITH_CUDA)
  return lang::CUDADriver::get_instance_without_context().detected();
#else
  return false;
#endif
}

}  // namespace taichi
