// Virtual memory allocator for CPU/GPU

#if defined(TI_WITH_CUDA)
#include "taichi/backends/cuda/cuda_driver.h"
#endif
#include "taichi/lang_util.h"
#include "taichi/system/unified_allocator.h"
#include "taichi/system/virtual_memory.h"
#include "taichi/system/timer.h"
#include <string>

TLANG_NAMESPACE_BEGIN

UnifiedAllocator::UnifiedAllocator(std::size_t size, Arch arch)
    : size(size), arch_(arch) {
  auto t = Time::get_time();
  if (arch_ == Arch::cuda) {
    // CUDA gets stuck when
    //  - kernel A requests memory
    //  - the memory allocator trys to allocate memory (and get stuck at
    //  cudaMallocManaged for some reason)
    //  - kernel B is getting loaded via cuModuleLoadDataEx (and get stuck for
    //  some reason)
    // So we need a mutex here...
    TI_TRACE("Allocating unified (CPU+GPU) address space of size {} MB",
             size / 1024 / 1024);
#if defined(TI_WITH_CUDA)
    // std::lock_guard<std::mutex> _(cuda_context->lock);
    CUDADriver::get_instance().malloc_managed(&_cuda_data, size,
                                              CU_MEM_ATTACH_GLOBAL);
    if (_cuda_data == nullptr) {
      TI_ERROR("CUDA memory allocation failed.");
    }
#if !defined(TI_ARCH_ARM) && !defined(TI_PLATFORM_WINDOWS)
    // Assuming ARM devices have shared CPU/GPU memory and do no support
    // memAdvise; CUDA on Windows has limited support for unified memory
    CUDADriver::get_instance().mem_advise.call_with_warning(
        _cuda_data, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, 0);
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
    TI_NOT_IMPLEMENTED
#endif
  } else {
    TI_TRACE("Allocating virtual address space of size {} MB",
             size / 1024 / 1024);
    cpu_vm = std::make_unique<VirtualMemoryAllocator>(size);
    data = (uint8 *)cpu_vm->ptr;
  }
  TI_ASSERT(data != nullptr);
  TI_ASSERT(uint64(data) % 4096 == 0);

  head = data;
  tail = head + size;
  TI_TRACE("Memory allocated. Allocation time = {:.3} s", Time::get_time() - t);
}

taichi::lang::UnifiedAllocator::~UnifiedAllocator() {
  if (!initialized()) {
    return;
  }
  if (arch_ == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().mem_free(_cuda_data);
#else
    TI_ERROR("No CUDA support");
#endif
  }
}

void taichi::lang::UnifiedAllocator::memset(unsigned char val) {
  std::memset(data, val, size);
}

TLANG_NAMESPACE_END
