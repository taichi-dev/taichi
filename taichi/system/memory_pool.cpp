#include "memory_pool.h"
#include "taichi/system/timer.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_device.h"

#include "taichi/common/core.h"

#if defined(TI_PLATFORM_UNIX)
#include <sys/mman.h>
#else
#include "taichi/platform/windows/windows.h"
#endif

namespace taichi::lang {

MemoryPool &MemoryPool::get_instance(Arch arch) {
  if (!arch_is_cuda(arch) && !arch_is_cpu(arch)) {
    arch = host_arch();
  }

  if (arch_is_cuda(arch)) {
    static MemoryPool *cuda_memory_pool = new MemoryPool(arch);
    return *cuda_memory_pool;
  }

  static MemoryPool *cpu_memory_pool = new MemoryPool(arch);
  return *cpu_memory_pool;
}

// In the future we wish to move the MemoryPool inside each Device
// so that the memory allocated from each Device can be used as-is.
MemoryPool::MemoryPool(Arch arch) : arch_(arch) {
  TI_TRACE("Memory pool created. Default buffer size per allocator = {} MB",
           default_allocator_size / 1024 / 1024);
  // TODO: initialize allocator according to arch
}

void *MemoryPool::allocate(std::size_t size,
                           std::size_t alignment,
                           bool releasable) {
  std::lock_guard<std::mutex> _(mut_allocators);
  void *ret = nullptr;

  // TODO: refactor this part to allocator->allocate(size, alignment)
  if (arch_is_cpu(arch_)) {
    if (releasable) {
      // For UnifiedAllocator, we have to make it exclusive to make sure it's
      // releasable, usually used for small memory allocations so it's easy to
      // alloc - release
      allocators.emplace_back(std::make_unique<UnifiedAllocator>(
          size, arch_, true /* is_exclusive */));
      ret = allocators.back()->allocate(size, alignment);

    } else if (!allocators.empty()) {
      ret = allocators.back()->allocate(size, alignment);
    }

    if (!ret) {
      // allocation have failed
      auto new_buffer_size = std::max(size, default_allocator_size);
      allocators.emplace_back(
          std::make_unique<UnifiedAllocator>(new_buffer_size, arch_));
      ret = allocators.back()->allocate(size, alignment);
    }
    TI_ASSERT(ret);
  } else {
    TI_NOT_IMPLEMENTED;
  }

  return ret;
}

void MemoryPool::release(std::size_t size, void *ptr) {
  std::lock_guard<std::mutex> _(mut_allocators);

  for (auto &allocator : allocators) {
    if (allocator->is_releasable((uint64_t *)ptr)) {
      allocator->release(size, (uint64_t *)ptr);
      return;
    }
  }
}

void *MemoryPool::allocate_raw_memory(std::size_t size) {
  /*
    Be aware that this methods is not protected by the mutex.

    allocate_raw_memory() is designed to be a private method, and
    should only be called by its Allocators friends.

    The caller ensures that no other thread is accessing the memory pool
    when calling this method.
  */

  void *ptr = nullptr;
  if (arch_is_cpu(arch_)) {
#if defined(TI_PLATFORM_UNIX)
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TI_ERROR_IF(ptr == MAP_FAILED, "Virtual memory allocation ({} B) failed.",
                size);
#else
    MEMORYSTATUSEX stat;
    stat.dwLength = sizeof(stat);
    GlobalMemoryStatusEx(&stat);
    if (stat.ullAvailVirtual < size) {
      TI_P(stat.ullAvailVirtual);
      TI_P(size);
      TI_ERROR("Insufficient virtual memory space");
    }
    ptr = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    TI_ERROR_IF(ptr == nullptr, "Virtual memory allocation ({} B) failed.",
                size);
#endif
    TI_ERROR_IF(((uint64_t)ptr) % page_size != 0,
                "Allocated address ({:}) is not aligned by page size {}", ptr,
                page_size);
  } else {
    TI_NOT_IMPLEMENTED;
  }

  if (raw_memory_chunks_.count(ptr)) {
    TI_ERROR("Memory address ({:}) is already allocated", ptr);
  }

  raw_memory_chunks_[ptr] = size;
  return ptr;
}

void MemoryPool::deallocate_raw_memory(void *ptr) {
  /*
    Be aware that this methods is not protected by the mutex.

    deallocate_raw_memory() is designed to be a private method, and
    should only be called by its Allocators friends.

    The caller ensures that no other thread is accessing the memory pool
    when calling this method.
  */
  if (!raw_memory_chunks_.count(ptr)) {
    TI_ERROR("Memory address ({:}) is not allocated", ptr);
  }

  std::size_t size = raw_memory_chunks_[ptr];
#if defined(TI_PLATFORM_UNIX)
  if (munmap(ptr, size) != 0)
#else
  // https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualfree
  // According to MS Doc: size must be when using MEM_RELEASE
  if (!VirtualFree(ptr, 0, MEM_RELEASE))
#endif
    TI_ERROR("Failed to free virtual memory ({} B)", size);

  raw_memory_chunks_.erase(ptr);
}

void MemoryPool::reset() {
  std::lock_guard<std::mutex> _(mut_allocators);
  allocators.clear();

  const auto ptr_map_copied = raw_memory_chunks_;
  for (auto &ptr : ptr_map_copied) {
    deallocate_raw_memory(ptr.first);
  }
}

MemoryPool::~MemoryPool() {
  reset();
}

}  // namespace taichi::lang
