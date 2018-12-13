#include "tlang.h"
#include <cuda_runtime.h>

TC_NAMESPACE_BEGIN

namespace Tlang {

AlignedAllocator::AlignedAllocator(std::size_t size, Device device)
    : device(device) {
  if (device == Device::cpu) {
    _data.resize(size + 4096);
    data = _data.data();
  } else {
    TC_ASSERT(device == Device::gpu);
    cudaMallocManaged(&_cuda_data, size + 4096);
    data = _cuda_data;
  }
  auto p = reinterpret_cast<uint64>(data);
  data = (void *)(p + (4096 - p % 4096));
}

AlignedAllocator::~AlignedAllocator() {
  if (!initialized()) {
    return;
  }
  if (device == Device::cpu) {
  } else {
    TC_ASSERT(device == Device::gpu);
    cudaFree(_cuda_data);
  }
}
}

TC_NAMESPACE_END
