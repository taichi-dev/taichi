#pragma once

#include <taichi/common/util.h>

#if defined(TC_PLATFORM_UNIX)
#include <sys/mman.h>
#else
#include <windows.h>
#endif

TC_NAMESPACE_BEGIN

// Cross-platform virtual memory allocator
class VirtualMemoryAllocator {
 public:
  static constexpr size_t page_size = (1 << 12);  // 4 KB page size by default
  void *ptr;
  size_t size;
  explicit VirtualMemoryAllocator(size_t size) : size(size) {
// http://pages.cs.wisc.edu/~sifakis/papers/SPGrid.pdf Sec 3.1
#if defined(TC_PLATFORM_UNIX)
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    TC_ERROR_IF(ptr == MAP_FAILED, "Virtual memory allocation ({} B) failed.",
                size);
#else
    ptr = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    TC_ERROR_IF(ptr == nullptr, "Virtual memory allocation ({} B) failed.",
                size);
#endif
    TC_ERROR_IF(((uint64_t)ptr) % page_size != 0,
                "Allocated address ({:}) is not aligned by page size {}", ptr,
                page_size);
  }

  ~VirtualMemoryAllocator() {
#if defined(TC_PLATFORM_UNIX)
    if (munmap(ptr, size) != 0)
#else
    if (!VirtualFree(ptr, size, MEM_RELEASE))
#endif
      TC_ERROR("Failed to free virtual memory ({} B)", size);
  }
};

float64 get_memory_usage_gb(int pid = -1);
uint64 get_memory_usage(int pid = -1);

#define TC_MEMORY_USAGE(name) \
  TC_WARN("Memory Usage [{}] = {:.2f} GB", name, get_memory_usage_gb());

TC_NAMESPACE_END
