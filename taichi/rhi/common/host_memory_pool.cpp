#include "taichi/rhi/common/host_memory_pool.h"

#include <memory>

#if defined(TI_PLATFORM_UNIX)
#include <sys/mman.h>
#else
#include "taichi/platform/windows/windows.h"
#endif

namespace taichi::lang {

HostMemoryPool::HostMemoryPool() {
  allocator_ = std::unique_ptr<UnifiedAllocator>(new UnifiedAllocator());

  TI_TRACE("Memory pool created. Default buffer size per allocator = {} MB",
           UnifiedAllocator::default_allocator_size / 1024 / 1024);
}

void *HostMemoryPool::allocate(std::size_t size,
                               std::size_t alignment,
                               bool exclusive) {
  std::lock_guard<std::mutex> _(mut_allocation_);

  if (!allocator_) {
    TI_ERROR("Memory pool is already destroyed");
  }
  void *ret = allocator_->allocate(size, alignment, exclusive);
  return ret;
}

void HostMemoryPool::release(std::size_t size, void *ptr) {
  std::lock_guard<std::mutex> _(mut_allocation_);

  if (!allocator_) {
    TI_ERROR("Memory pool is already destroyed");
  }

  if (allocator_->release(size, ptr)) {
    if (dynamic_cast<UnifiedAllocator *>(allocator_.get())) {
      deallocate_raw_memory(ptr);  // release raw memory as well
    }
  }
}

void *HostMemoryPool::allocate_raw_memory(std::size_t size) {
  /*
    Be aware that this methods is not protected by the mutex.

    allocate_raw_memory() is designed to be a private method, and
    should only be called by its Allocators friends.

    The caller ensures that no other thread is accessing the memory pool
    when calling this method.
  */

  void *ptr = nullptr;
#if defined(TI_PLATFORM_UNIX)
  ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
             -1, 0);
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
  TI_ERROR_IF(ptr == nullptr, "Virtual memory allocation ({} B) failed.", size);
#endif
  TI_ERROR_IF(((uint64_t)ptr) % page_size != 0,
              "Allocated address ({:}) is not aligned by page size {}", ptr,
              page_size);

  if (raw_memory_chunks_.count(ptr)) {
    TI_ERROR("Memory address ({:}) is already allocated", ptr);
  }

  raw_memory_chunks_[ptr] = size;
  return ptr;
}

void HostMemoryPool::deallocate_raw_memory(void *ptr) {
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

void HostMemoryPool::reset() {
  std::lock_guard<std::mutex> _(mut_allocation_);
  allocator_ = std::unique_ptr<UnifiedAllocator>(new UnifiedAllocator());

  const auto ptr_map_copied = raw_memory_chunks_;
  for (auto &ptr : ptr_map_copied) {
    deallocate_raw_memory(ptr.first);
  }
}

HostMemoryPool::~HostMemoryPool() {
  reset();
}

const size_t HostMemoryPool::page_size{1 << 12};  // 4 KB page size by default

HostMemoryPool &HostMemoryPool::get_instance() {
  static HostMemoryPool *memory_pool = new HostMemoryPool();
  return *memory_pool;
}

}  // namespace taichi::lang
