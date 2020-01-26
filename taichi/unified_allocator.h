#pragma once
#include "common.h"
#include <mutex>
#include <vector>
#include <memory>

namespace taichi {
class VirtualMemoryAllocator;
}

TLANG_NAMESPACE_BEGIN

class UnifiedAllocator;

extern UnifiedAllocator *&allocator();
extern UnifiedAllocator *allocator_instance;

// This class can only have one instance
class UnifiedAllocator {
  std::unique_ptr<VirtualMemoryAllocator> cpu_vm;
#if defined(TLANG_WITH_CUDA)
  void *_cuda_data;
#endif
  std::size_t size;
  bool gpu;

  // put these two on the unified memory so that GPU can have access
 public:
  uint8 *data;
  uint8 *head;
  uint8 *tail;
  std::mutex lock;

 public:
  UnifiedAllocator(bool gpu);

  ~UnifiedAllocator();

  void *alloc(std::size_t size, std::size_t alignment) {
    std::lock_guard<std::mutex> _(lock);
    auto ret =
        head + alignment - 1 - ((std::size_t)head + alignment - 1) % alignment;
    head = ret + size;
    if (head > tail) {
      // allocation failed
      return nullptr;
    } else {
      // success
      TC_ASSERT((std::size_t)ret % alignment == 0);
      return ret;
    }
  }

  void memset(unsigned char val);

  bool initialized() const {
    return data != nullptr;
  }

  UnifiedAllocator operator=(const UnifiedAllocator &) = delete;
};

inline void *allocate(std::size_t size, int alignment = 1) {
  return allocator()->alloc(size, alignment);
}

TLANG_NAMESPACE_END
