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
  void *_cuda_data{};
#endif
  std::size_t size{};
  bool gpu{};

  // put these two on the unified memory so that GPU can have access
 public:
  void *data;
  void **head{};
  void **tail{};
  int gpu_error_code;
  std::mutex lock;

 public:
  UnifiedAllocator();

  UnifiedAllocator(std::size_t size, bool gpu);

  ~UnifiedAllocator();

  void *alloc(std::size_t size, int alignment) {
    std::lock_guard<std::mutex> _(lock);
    auto ret = (char *)(*head) + alignment - 1 -
               ((std::size_t)(char *)(*head) + alignment - 1) % alignment;
    *head = (char *)ret + size;
    return ret;
  }

  void memset(unsigned char val);

  bool initialized() const {
    return data != nullptr;
  }

  UnifiedAllocator operator=(const UnifiedAllocator &) = delete;

  static void create(bool gpu);

  static void free();
};

TC_FORCE_INLINE void *allocate(std::size_t size,
                                                   int alignment = 1) {
#if __CUDA_ARCH__
  auto addr = allocator()->alloc_gpu(*device_head, size, alignment);
#else
  auto addr = allocator()->alloc(size, alignment);
#endif
  return addr;
}


TLANG_NAMESPACE_END
