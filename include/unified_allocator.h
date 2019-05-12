#pragma once
#include "common.h"
#include <mutex>
#include <vector>

TLANG_NAMESPACE_BEGIN

class UnifiedAllocator;

UnifiedAllocator *&allocator();

class UnifiedAllocator {
  std::vector<char> _data;
  void *_cuda_data{};
  std::size_t size;
  bool gpu;

  // put these two on the unified memroy so that GPU can have access
 public:
  void *data;
  void **head;
  void **tail;
  int gpu_error_code;
  std::mutex lock;

 public:
  UnifiedAllocator() {
    data = nullptr;
    gpu_error_code = 0;
  }

  UnifiedAllocator(std::size_t size, bool gpu);

#if defined(TLANG_GPU)
  __device__ static void *alloc_gpu(void *&head, int size, int alignment = 1) {
    if (alignment == 1) {
      return (void *)atomicAdd(reinterpret_cast<unsigned long long *>(&head),
                               size);
    } else {
      TC_ASSERT(false); // aligned allocation is not supported on GPU
      return nullptr;
    }
  }
#endif

  __host__ void *alloc(std::size_t size, int alignment) {
    std::lock_guard<std::mutex> _(lock);
    auto ret = (char *)(*head) + alignment - 1 -
               ((unsigned long)(char *)(*head) + alignment - 1) % alignment;
    *head = (char *)ret + size;
    return ret;
  }

  ~UnifiedAllocator();

  void memset(unsigned char val);

  bool initialized() const {
    return data != nullptr;
  }

  UnifiedAllocator operator=(const UnifiedAllocator &) = delete;

  static void create();

  static void free();
};

TC_FORCE_INLINE __host__ __device__ void *allocate(std::size_t size,
                                                   int alignment = 1) {
#if __CUDA_ARCH__
  auto addr = allocator()->alloc_gpu(*device_head, size, alignment);
#else
  auto addr = allocator()->alloc(size, alignment);
#endif
  return addr;
}

template <typename T>
TC_FORCE_INLINE __host__ __device__ T *allocate() {
  auto addr = allocate(sizeof(T));
  return new (addr) T();
}

template <typename T, typename... Args>
__host__ T *create_unified(Args &&... args) {
  auto addr = allocate(sizeof(T));
  return new (addr) T(std::forward<Args>(args)...);
}

TLANG_NAMESPACE_END
