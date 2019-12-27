// Virtual memory allocator for CPU/GPU

#if defined(CUDA_FOUND)
#include "cuda_utils.h"
#endif
#include "tlang_util.h"
#include <taichi/unified_allocator.h>
#include <taichi/system/virtual_memory.h>
#include <string>

TLANG_NAMESPACE_BEGIN

UnifiedAllocator *allocator_instance = nullptr;
UnifiedAllocator *&allocator() {
  return allocator_instance;
}

taichi::Tlang::UnifiedAllocator::UnifiedAllocator(std::size_t size, bool gpu)
    : size(size), gpu(gpu) {
  size += 4096;
  if (gpu) {
#if defined(CUDA_FOUND)
    check_cuda_errors(cudaMallocManaged(&_cuda_data, size + 4096));
    if (_cuda_data == nullptr) {
      TC_ERROR("GPU memory allocation failed.");
    }
    check_cuda_errors(cudaMemAdvise(_cuda_data, size + 4096,
                                    cudaMemAdviseSetPreferredLocation, 0));
    // http://on-demand.gputechconf.com/gtc/2017/presentation/s7285-nikolay-sakharnykh-unified-memory-on-pascal-and-volta.pdf
    /*
    cudaMemAdvise(_cuda_data, size + 4096, cudaMemAdviseSetReadMostly,
                  cudaCpuDeviceId);
    cudaMemAdvise(_cuda_data, size + 4096, cudaMemAdviseSetAccessedBy,
                  0);
                  */
    data = _cuda_data;
#else
    TC_NOT_IMPLEMENTED
#endif
  } else {
    cpu_vm = std::make_unique<VirtualMemoryAllocator>(size);
    data = cpu_vm->ptr;
  }
  TC_ASSERT(data != nullptr);
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
    check_cuda_errors(cudaFree(_cuda_data));
#else
    TC_ERROR("No CUDA support");
#endif
  }
}

void taichi::Tlang::UnifiedAllocator::create(bool gpu) {
  TC_ASSERT(allocator() == nullptr);
  void *dst;
  if (gpu) {
#if defined(CUDA_FOUND)
    gpu = true;
    check_cuda_errors(cudaMallocManaged(&dst, sizeof(UnifiedAllocator)));
#else
    TC_NOT_IMPLEMENTED
#endif
  } else {
    dst = std::malloc(sizeof(UnifiedAllocator));
  }
#if !defined(TC_PLATFORM_WINDOWS)
  allocator() = new (dst) UnifiedAllocator(1LL << 44, gpu);
#else
  std::size_t phys_mem_size;
  GetPhysicallyInstalledSystemMemory(&phys_mem_size);  // KB
  phys_mem_size /= 1024;                               // MB
  TC_INFO("Physical memory size size {} MB", phys_mem_size);
  auto virtual_mem_to_allocate = (phys_mem_size << 20) / 4;
  TC_INFO("Allocating virtual memory pool (size = {} MB)",
          virtual_mem_to_allocate / 1024 / 1024);
  allocator() = new (dst) UnifiedAllocator(virtual_mem_to_allocate, gpu);
#endif
}

void taichi::Tlang::UnifiedAllocator::free() {
  (*allocator()).~UnifiedAllocator();
  if (allocator()->gpu) {
#if defined(CUDA_FOUND)
    check_cuda_errors(cudaFree(allocator()));
#else
    TC_NOT_IMPLEMENTED
#endif
  } else {
    std::free(allocator());
  }
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

extern "C" void *taichi_allocate_aligned(std::size_t size, int alignment) {
  return taichi::Tlang::allocate(size, alignment);
}
