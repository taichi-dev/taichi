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

UnifiedAllocator::UnifiedAllocator(bool gpu) : gpu(gpu) {
#if !defined(TC_PLATFORM_WINDOWS)

#if defined(TI_ARCH_ARM)
  // Try to allocate only 2GB RAM on ARM devices such as Jetson nano
  std::size_t size = 1LL << 31;
#else
  std::size_t size = 1LL << 44;
#endif

#else
  std::size_t phys_mem_size;
  if (GetPhysicallyInstalledSystemMemory(&phys_mem_size)) {  // KB
    phys_mem_size /= 1024;                                   // MB
    TC_INFO("Physical memory size {} MB", phys_mem_size);
  } else {
    auto err = GetLastError();
    /* Error Codes:
     * https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes
     */
    TC_WARN("Unable to get physical memory size [ Win32 Error Code {} ].", err);
    phys_mem_size = 4096 * 4;  // allocate 4 GB later
  }
  auto virtual_mem_to_allocate = (phys_mem_size << 20) / 4;
  TC_INFO("Allocating virtual memory pool (size = {} MB)",
          virtual_mem_to_allocate / 1024 / 1024);
  std::size_t size = virtual_mem_to_allocate;
#endif
  this->size = size;
  if (gpu) {
#if defined(CUDA_FOUND)
    check_cuda_errors(cudaMallocManaged(&_cuda_data, size));
    if (_cuda_data == nullptr) {
      TC_ERROR("GPU memory allocation failed.");
    }
#if !defined(TI_ARCH_ARM)
    // Assuming ARM devices have shared CPU/GPU memory and do no support
    // memAdvise
    check_cuda_errors(
        cudaMemAdvise(_cuda_data, size, cudaMemAdviseSetPreferredLocation, 0));
#endif
    // http://on-demand.gputechconf.com/gtc/2017/presentation/s7285-nikolay-sakharnykh-unified-memory-on-pascal-and-volta.pdf
    /*
    cudaMemAdvise(_cuda_data, size, cudaMemAdviseSetReadMostly,
                  cudaCpuDeviceId);
    cudaMemAdvise(_cuda_data, size, cudaMemAdviseSetAccessedBy,
                  0);
                  */
    data = (uint8 *)_cuda_data;
#else
    TC_NOT_IMPLEMENTED
#endif
  } else {
    cpu_vm = std::make_unique<VirtualMemoryAllocator>(size);
    data = (uint8 *)cpu_vm->ptr;
  }
  TC_ASSERT(data != nullptr);
  TC_ASSERT(uint64(data) % 4096 == 0);

  head = data;
  tail = head + size;
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
  allocator() = new UnifiedAllocator(gpu);
}

void taichi::Tlang::UnifiedAllocator::free() {
  (*allocator()).~UnifiedAllocator();
  allocator() = nullptr;
}

void taichi::Tlang::UnifiedAllocator::memset(unsigned char val) {
  std::memset(data, val, size);
}

UnifiedAllocator::UnifiedAllocator() {
  data = nullptr;
}

TLANG_NAMESPACE_END

extern "C" void *taichi_allocate_aligned(std::size_t size, int alignment) {
  return taichi::Tlang::allocate(size, alignment);
}
