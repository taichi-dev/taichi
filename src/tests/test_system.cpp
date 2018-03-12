/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/util.h>
#include <taichi/testing.h>
#include <taichi/math/svd.h>
#include <taichi/math/eigen.h>

TC_NAMESPACE_BEGIN

constexpr size_t page_size = (1 << 12);  // 4 KB page size by default

#if defined(TC_PLATFORM_UNIX)
#include <sys/mman.h>
#else
#include <windows.h>
#endif

// Cross-platform virtual memory allocator
class VirtualMemoryAllocator {
public:
  void *ptr;
  size_t size;
  explicit VirtualMemoryAllocator(size_t size) : size(size) {
// http://pages.cs.wisc.edu/~sifakis/papers/SPGrid.pdf Sec 3.1
#if defined(TC_PLATFORM_UNIX)
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
#else
    ptr = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT,
                       PAGE_READWRITE);
#endif
    TC_ERROR_IF(ptr == MAP_FAILED, "Virtual memory allocation ({} B) failed.",
                size);
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

inline uint64 rand_int64() {
  return ((uint64)rand_int() << 32) + rand_int();
}

TC_TEST("Virtual Memory") {
  for (int i = 0; i < 3; i++) {
    // Allocate 1 TB of virtual memory
    std::size_t size = 1LL << 40;
    VirtualMemoryAllocator vm(size);
    // Touch 512 MB (1 << 29 B)
    for (int j = 0; j < (1 << 29) / page_size; j++) {
      void *target = vm.ptr + rand_int64() % size;
      uint8 val = *(uint8 *)target;
      CHECK(val == 0);
    }
  }

}

TC_NAMESPACE_END
