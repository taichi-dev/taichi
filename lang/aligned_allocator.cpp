#include "tlang.h"
#include <cuda_runtime.h>

TC_NAMESPACE_BEGIN

namespace Tlang {

AlignedAllocator::AlignedAllocator(std::size_t size, Device device)
    : device(device) {
  if (device == Device::cpu) {
    _data.resize(size + 4096);
    auto p = reinterpret_cast<uint64>(_data.data());
    data = (void *)(p + (4096 - p % 4096));
  } else {
    TC_ASSERT(device == Device::gpu);
    cudaMallocManaged(&data, size);
    TC_ASSERT(uint64(data) % 4096 == 0);
  }
}

AlignedAllocator::~AlignedAllocator() {
  if (!initialized()) {
    return;
  }
  if (device == Device::cpu) {
  } else {
    TC_ASSERT(device == Device::gpu);
    cudaFree(data);
  }
}

}

TC_NAMESPACE_END
