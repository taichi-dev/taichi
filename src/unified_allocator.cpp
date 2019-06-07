#if defined(CUDA_FOUND)
#include <cuda_runtime.h>
#endif
#include "util.h"
#include "../include/unified_allocator.h"
#include <taichi/system/virtual_memory.h>
#include <string>

TLANG_NAMESPACE_BEGIN

UnifiedAllocator *&allocator() {
  static UnifiedAllocator *instance = nullptr;
  return instance;
}

taichi::Tlang::UnifiedAllocator::UnifiedAllocator(std::size_t size, bool gpu)
    : size(size), gpu(gpu) {
  size += 4096;
  if (!gpu) {
    _data.resize(size + 4096);
    data = _data.data();
  } else {
#if defined(CUDA_FOUND)
    cudaMallocManaged(&_cuda_data, size + 4096);
    cudaMemAdvise(_cuda_data, size + 4096, cudaMemAdviseSetPreferredLocation,
                  0);
    // http://on-demand.gputechconf.com/gtc/2017/presentation/s7285-nikolay-sakharnykh-unified-memory-on-pascal-and-volta.pdf
    /*
    cudaMemAdvise(_cuda_data, size + 4096, cudaMemAdviseSetReadMostly,
                  cudaCpuDeviceId);
    cudaMemAdvise(_cuda_data, size + 4096, cudaMemAdviseSetAccessedBy,
                  0);
                  */
    data = _cuda_data;
#else
    cpu_vm = std::make_unique<VirtualMemoryAllocator>(size);
    data = cpu_vm->ptr;
#endif
  }
  auto p = reinterpret_cast<uint64>(data);
  data = (void *)(p + (4096 - p % 4096));

  // allocate head/tail ptrs on unified memory
  head = (void **)data;
  tail = (void **)((char *)data + sizeof(void *));

  data = (char *)data + 4096;
  *head = data;
  *tail = (void *)(((char *)head) + size);
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
#if defined(CUDA_FOUND)
  cudaMallocManaged(&dst, sizeof(UnifiedAllocator));
#else
  dst = std::malloc(sizeof(UnifiedAllocator));
#endif
  allocator() = new (dst) UnifiedAllocator(1LL << 40, true);
}

void taichi::Tlang::UnifiedAllocator::free() {
#if defined(CUDA_FOUND)
  cudaFree(allocator());
#else
  std::free(allocator());
#endif
  allocator() = nullptr;
}

void taichi::Tlang::UnifiedAllocator::memset(unsigned char val) {
  std::memset(data, val, size);
}

UnifiedAllocator::UnifiedAllocator() {
  data = nullptr;
  gpu_error_code = 0;
}

TLANG_NAMESPACE_END
