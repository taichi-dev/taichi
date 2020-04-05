#pragma once
#include <mutex>
#include <vector>
#include <memory>

#include "taichi/program/arch.h"

namespace taichi {
class VirtualMemoryAllocator;
}

TLANG_NAMESPACE_BEGIN

// This class can only have one instance
class UnifiedAllocator {
  std::unique_ptr<VirtualMemoryAllocator> cpu_vm;
#if defined(TI_WITH_CUDA)
  void *_cuda_data;
#endif
  std::size_t size;
  Arch arch_;

  // put these two on the unified memory so that GPU can have access
 public:
  uint8 *data;
  uint8 *head;
  uint8 *tail;
  std::mutex lock;

 public:
  UnifiedAllocator(std::size_t size, Arch arch);

  ~UnifiedAllocator();

  void *allocate(std::size_t size, std::size_t alignment) {
    std::lock_guard<std::mutex> _(lock);
    auto ret =
        head + alignment - 1 - ((std::size_t)head + alignment - 1) % alignment;
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

  void memset(unsigned char val);

  bool initialized() const {
    return data != nullptr;
  }

  UnifiedAllocator operator=(const UnifiedAllocator &) = delete;
};

TLANG_NAMESPACE_END
