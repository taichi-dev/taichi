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

// In the future we wish to move the MemoryPool inside each Device
// so that the memory allocated from each Device can be used as-is.
MemoryPool::MemoryPool(Arch arch, Device *device)
    : arch_(arch), device_(device) {
  TI_TRACE("Memory pool created. Default buffer size per allocator = {} MB",
           default_allocator_size / 1024 / 1024);
}

void *MemoryPool::allocate(std::size_t size, std::size_t alignment) {
  std::lock_guard<std::mutex> _(mut_allocators);
  void *ret = nullptr;
  if (!allocators.empty()) {
    ret = allocators.back()->allocate(size, alignment);
  }
  if (!ret) {
    // allocation have failed
    auto new_buffer_size = std::max(size, default_allocator_size);
    allocators.emplace_back(
        std::make_unique<UnifiedAllocator>(new_buffer_size, arch_, device_));
    ret = allocators.back()->allocate(size, alignment);
  }
  TI_ASSERT(ret);
  return ret;
}

void *MemoryPool::allocate_raw_memory(std::size_t size, Arch arch) {
  std::lock_guard<std::mutex> _(mut_raw_alloc);
  void *ptr = nullptr;
  if (arch_is_cpu(arch)) {
// http://pages.cs.wisc.edu/~sifakis/papers/SPGrid.pdf Sec 3.1
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

  if (ptr_map_.count(ptr)) {
    TI_ERROR("Memory address ({:}) is already allocated", ptr);
  }

  ptr_map_[ptr] = size;
  return ptr;
}

void MemoryPool::deallocate_raw_memory(void *ptr) {
  if (!ptr_map_.count(ptr)) {
    TI_ERROR("Memory address ({:}) is not allocated", ptr);
  }

  std::size_t size = ptr_map_[ptr];
#if defined(TI_PLATFORM_UNIX)
  if (munmap(ptr, size) != 0)
#else
  // https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualfree
  // According to MS Doc: size must be when using MEM_RELEASE
  if (!VirtualFree(ptr, 0, MEM_RELEASE))
#endif
    TI_ERROR("Failed to free virtual memory ({} B)", size);

  ptr_map_.erase(ptr);
}

MemoryPool::~MemoryPool() {
  const auto ptr_map_copied = ptr_map_;

  for (auto &ptr : ptr_map_copied) {
    deallocate_raw_memory(ptr.first);
  }
}

}  // namespace taichi::lang
