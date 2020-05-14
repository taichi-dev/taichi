#pragma once

#include "taichi/common/core.h"

#if defined(TI_PLATFORM_UNIX)
#include <sys/mman.h>
#else
#include "taichi/platform/windows/windows.h"
#endif

TI_NAMESPACE_BEGIN

// Cross-platform virtual memory allocator
class VirtualMemoryAllocator {
 public:
  static constexpr size_t page_size = (1 << 12);  // 4 KB page size by default
  void *ptr;
  size_t size;
  explicit VirtualMemoryAllocator(size_t size) : size(size) {
// http://pages.cs.wisc.edu/~sifakis/papers/SPGrid.pdf Sec 3.1
#if defined(TI_PLATFORM_UNIX)
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
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
  }

  ~VirtualMemoryAllocator() {
#if defined(TI_PLATFORM_UNIX)
    if (munmap(ptr, size) != 0)
#else
    // https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualfree
    // According to MS Doc: size must be when using MEM_RELEASE
    if (!VirtualFree(ptr, 0, MEM_RELEASE))
#endif
      TI_ERROR("Failed to free virtual memory ({} B)", size);
  }
};

float64 get_memory_usage_gb(int pid = -1);
uint64 get_memory_usage(int pid = -1);

#define TI_MEMORY_USAGE(name) \
  TI_DEBUG("Memory Usage [{}] = {:.2f} GB", name, get_memory_usage_gb());

TI_NAMESPACE_END
