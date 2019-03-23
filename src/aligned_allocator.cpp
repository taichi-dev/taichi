#include "tlang.h"
#if defined(CUDA_FOUND)
#include <cuda_runtime.h>
#endif

TC_NAMESPACE_BEGIN

namespace Tlang {

AlignedAllocator::AlignedAllocator(std::size_t size, Device device)
    : size(size), device(device) {
  if (device == Device::cpu) {
    _data.resize(size + 4096);
    data = _data.data();
  } else {
#if defined(CUDA_FOUND)
    TC_ASSERT(device == Device::gpu);
    cudaMallocManaged(&_cuda_data, size + 4096);
    data = _cuda_data;
#else
    TC_ERROR("No CUDA support");
#endif
  }
  auto p = reinterpret_cast<uint64>(data);
  data = (void *)(p + (4096 - p % 4096));
  memset(0);
}

AlignedAllocator::~AlignedAllocator() {
  if (!initialized()) {
    return;
  }
  if (device == Device::cpu) {
  } else {
    TC_ASSERT(device == Device::gpu);
#if defined(CUDA_FOUND)
    cudaFree(_cuda_data);
#else
    TC_ERROR("No CUDA support");
#endif
  }
}
}  // namespace Tlang

TC_NAMESPACE_END
