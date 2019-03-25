#include "../include/unified_allocator.h"
#include "util.h"
#include <string>
#if defined(CUDA_FOUND)
#include <cuda_runtime.h>
#endif

TLANG_NAMESPACE_BEGIN

UnifiedAllocator *&allocator() {
  static UnifiedAllocator *instance = nullptr;
  return instance;
}

taichi::Tlang::UnifiedAllocator::UnifiedAllocator(std::size_t size, bool gpu)
    : size(size), gpu(gpu) {
  if (!gpu) {
    _data.resize(size + 4096);
    data = _data.data();
  } else {
#if defined(CUDA_FOUND)
    cudaMallocManaged(&_cuda_data, size + 4096);
    data = _cuda_data;
#else
    static_assert(false, "implement mmap on CPU for memset..");
    TC_ERROR("No CUDA support");
#endif
  }
  auto p = reinterpret_cast<uint64>(data);
  data = (void *)(p + (4096 - p % 4096));
  head = data;
  tail = (void *)(((char *)head) + size);
  // memset(0);
}

taichi::Tlang::UnifiedAllocator::~UnifiedAllocator() {
  if (!initialized()) {
    return;
  }
  if (gpu) {
#if defined(CUDA_FOUND)
    cudaFree(_cuda_data);
#else
    TC_ERROR("No CUDA support");
#endif
  }
}

void taichi::Tlang::UnifiedAllocator::create() {
  TC_ASSERT(allocator() == nullptr);
  void *dst;
  cudaMallocManaged(&dst, sizeof(UnifiedAllocator));
  allocator() = new (dst) UnifiedAllocator(1LL << 40, true);
}

void taichi::Tlang::UnifiedAllocator::free() {
  cudaFree(allocator());
  allocator() = nullptr;
}

void taichi::Tlang::UnifiedAllocator::memset(unsigned char val) {
  std::memset(data, val, size);
}

TLANG_NAMESPACE_END
