// Virtual memory allocator for CPU/GPU

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"

#endif
#include "taichi/util/lang_util.h"
#include "taichi/system/unified_allocator.h"
#include "taichi/system/virtual_memory.h"
#include "taichi/system/memory_pool.h"
#include "taichi/system/timer.h"
#include "taichi/rhi/cpu/cpu_device.h"
#include <string>

namespace taichi::lang {

UnifiedAllocator::UnifiedAllocator(std::size_t size, Arch arch) : size_(size) {
  auto t = Time::get_time();

  TI_TRACE("Allocating virtual address space of size {} MB",
           size / 1024 / 1024);
  void *ptr = MemoryPool::get_instance(arch).allocate_raw_memory(size);
  data = (uint8 *)ptr;

  TI_ASSERT(data != nullptr);
  TI_ASSERT(uint64(data) % 4096 == 0);

  head = data;
  tail = head + size;
  TI_TRACE("Memory allocated. Allocation time = {:.3} s", Time::get_time() - t);
}

void *UnifiedAllocator::allocate(std::size_t size, std::size_t alignment) {
  // UnifiedAllocator never reuses the previously allocated memory
  // just move the head forward util depleting all the free memory

  // Note: put mutex on MemoryPool instead of Allocator, since Allocators are
  // transparent to user code
  auto ret =
      head + alignment - 1 - ((std::size_t)head + alignment - 1) % alignment;
  TI_TRACE("UM [data={}] allocate() request={} remain={}", (intptr_t)data, size,
           (tail - head));
  head = ret + size;
  if (head > tail) {
    // allocation failed
    return nullptr;
  } else {
    // success
    TI_ASSERT((std::size_t)ret % alignment == 0);
    return ret;
  }
}

void UnifiedAllocator::release(size_t sz, uint64_t *ptr) {
  // UnifiedAllocator never reuses the previously allocated memory
  // therefore there's nothing to do here
}

taichi::lang::UnifiedAllocator::~UnifiedAllocator() {
  // Raw memory is always fully managed by MemoryPool once allocated
  // There's nothing to do here
}

void taichi::lang::UnifiedAllocator::memset(unsigned char val) {
  std::memset(data, val, size_);
}

}  // namespace taichi::lang
