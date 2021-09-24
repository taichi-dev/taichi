// Virtual memory allocator for CPU/GPU

#if defined(TI_WITH_CUDA)
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/backends/cuda/cuda_device.h"

#endif
#include "taichi/lang_util.h"
#include "taichi/system/unified_allocator.h"
#include "taichi/system/virtual_memory.h"
#include "taichi/system/timer.h"
#include "taichi/backends/cpu/cpu_device.h"
#include <string>

TLANG_NAMESPACE_BEGIN

UnifiedAllocator::UnifiedAllocator(std::size_t size, Arch arch, Device *device)
    : size(size), arch_(arch), device_(device) {
  auto t = Time::get_time();
  if (arch_ == Arch::cuda) {
    // CUDA gets stuck when
    //  - kernel A requests memory
    //  - the memory allocator trys to allocate memory (and get stuck at
    //  cudaMallocManaged for some reason)
    //  - kernel B is getting loaded via cuModuleLoadDataEx (and get stuck for
    //  some reason)
    // So we need a mutex here...
    // std::lock_guard<std::mutex> _(cuda_context->lock);
    TI_TRACE("Allocating unified (CPU+GPU) address space of size {} MB",
             size / 1024 / 1024);
#if defined(TI_WITH_CUDA)
    // This could be run on a host worker thread, so we have to set the context
    // before using any of the CUDA driver function call.
    auto _ = CUDAContext::get_instance().get_guard();

    Device::AllocParams alloc_params;
    alloc_params.size = size;
    alloc_params.host_read = true;
    alloc_params.host_write = true;

    cuda::CudaDevice *cuda_device = static_cast<cuda::CudaDevice *>(device);
    cuda_alloc = cuda_device->allocate_memory(alloc_params);
    _cuda_data = cuda_device->get_alloc_info(cuda_alloc).ptr;

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
    TI_TRACE("UM created, data={}", (intptr_t)data);
#else
    TI_NOT_IMPLEMENTED
#endif
  }
  // This is an intermediate state.
  // We will use memory pools to implement `Device::allocate_memory` soon.
  else if (arch_ == Arch::x64) {
    Device::AllocParams alloc_params;
    alloc_params.size = size;
    alloc_params.host_read = true;
    alloc_params.host_write = true;

    cpu::CpuDevice *cpu_device = static_cast<cpu::CpuDevice *>(device);
    alloc = cpu_device->allocate_memory(alloc_params);
    data = (uint8 *)cpu_device->get_alloc_info(alloc).ptr;
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
    cuda::CudaDevice *cuda_device = static_cast<cuda::CudaDevice *>(device_);
    cuda_device->dealloc_memory(cuda_alloc);
#else
    TI_ERROR("No CUDA support");
#endif
  } else if (arch_ == Arch::x64) {
    cpu::CpuDevice *cpu_device = static_cast<cpu::CpuDevice *>(device_);
    cpu_device->dealloc_memory(alloc);
  }
}

void taichi::lang::UnifiedAllocator::memset(unsigned char val) {
  std::memset(data, val, size);
}

TLANG_NAMESPACE_END
