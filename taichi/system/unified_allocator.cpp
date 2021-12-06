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
    : size_(size), arch_(arch), device_(device) {
  auto t = Time::get_time();
  if (arch_ == Arch::x64) {
#ifdef TI_WITH_LLVM
    Device::AllocParams alloc_params;
    alloc_params.size = size;
    alloc_params.host_read = true;
    alloc_params.host_write = true;

    cpu::CpuDevice *cpu_device = static_cast<cpu::CpuDevice *>(device);
    alloc = cpu_device->allocate_memory(alloc_params);
    data = (uint8 *)cpu_device->get_alloc_info(alloc).ptr;
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    TI_TRACE("Allocating virtual address space of size {} MB",
             size / 1024 / 1024);
    cpu_vm_ = std::make_unique<VirtualMemoryAllocator>(size);
    data = (uint8 *)cpu_vm_->ptr;
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
  if (arch_ == Arch::x64) {
    cpu::CpuDevice *cpu_device = static_cast<cpu::CpuDevice *>(device_);
    cpu_device->dealloc_memory(alloc);
  }
}

void taichi::lang::UnifiedAllocator::memset(unsigned char val) {
  std::memset(data, val, size_);
}

TLANG_NAMESPACE_END
