// Virtual memory allocator for CPU/GPU

#if defined(CUDA_FOUND)
#include "cuda_utils.h"
#endif
#include "tlang_util.h"
#include <taichi/unified_allocator.h>
#include <taichi/system/virtual_memory.h>
#include <string>
#include "backends/cuda_context.h"

TLANG_NAMESPACE_BEGIN

UnifiedAllocator::UnifiedAllocator(std::size_t size, Arch arch)
    : size(size), arch_(arch) {
  if (arch_ == Arch::cuda) {
    auto lock = cuda_context->get_lock_guard();
    // CUDA gets stuck when
    //  - kernel A requests memory
    //  - the memory allocator trys to allocate memory (and get stuck for some reason)
    //  - kernel B is launched (and never finish for some reason)
    // So we need a mutex.
    TC_TRACE("Allocating unified (CPU+GPU) address space of size {} MB",
             size / 1024 / 1024);
#if defined(CUDA_FOUND)
    check_cuda_errors(cudaMallocManaged(&_cuda_data, size));
    if (_cuda_data == nullptr) {
      TC_ERROR("GPU memory allocation failed.");
    }
#if !defined(TI_ARCH_ARM) && !defined(TC_PLATFORM_WINDOWS)
    // Assuming ARM devices have shared CPU/GPU memory and do no support
    // memAdvise; CUDA on Windows has limited support for unified memory
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
    TC_TRACE("Allocating virtual address space of size {} MB",
             size / 1024 / 1024);
    cpu_vm = std::make_unique<VirtualMemoryAllocator>(size);
    data = (uint8 *)cpu_vm->ptr;
  }
  TC_ASSERT(data != nullptr);
  TC_ASSERT(uint64(data) % 4096 == 0);

  head = data;
  tail = head + size;
  TC_TRACE("Memory allocated.");
}

taichi::Tlang::UnifiedAllocator::~UnifiedAllocator() {
  if (!initialized()) {
    return;
  }
  if (arch_ == Arch::cuda) {
#if defined(CUDA_FOUND)
    check_cuda_errors(cudaFree(_cuda_data));
#else
    TC_ERROR("No CUDA support");
#endif
  }
}

void taichi::Tlang::UnifiedAllocator::memset(unsigned char val) {
  std::memset(data, val, size);
}

TLANG_NAMESPACE_END
