#pragma once
#include "common.h"
#include <mutex>
#include <vector>

TLANG_NAMESPACE_BEGIN

class UnifiedAllocator;

UnifiedAllocator *&allocator();

class UnifiedAllocator {
  std::vector<char> _data;
  void *data;
  void *_cuda_data{};
  std::size_t size;
  bool gpu;

  // put these two on the unified memroy so that GPU can have access
 public:
  void **head;
  void **tail;

 public:
  UnifiedAllocator() {
    data = nullptr;
  }

  UnifiedAllocator(std::size_t size, bool gpu);

#if defined(TC_GPU)
  __device__ static void *alloc(void *&head, int size) {
    return (void *)atomicAdd(reinterpret_cast<unsigned long long *>(&head),
                             size);
  }
#else
  std::mutex lock;
  void *alloc(int size) {
    std::lock_guard<std::mutex> _(lock);
    auto ret = *head;
    *head = (char *)(*head) + size;
    return ret;
  }
#endif

  ~UnifiedAllocator();

  void memset(unsigned char val);

  bool initialized() const {
    return data != nullptr;
  }

  UnifiedAllocator operator=(const UnifiedAllocator &) = delete;

  static void create();

  static void free();
};

TLANG_NAMESPACE_END
