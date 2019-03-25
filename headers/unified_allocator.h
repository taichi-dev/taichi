#pragma once
#include <mutex>
#include <vector>

namespace taichi {
namespace Tlang {

class UnifiedAllocator;

UnifiedAllocator *&allocator();

class UnifiedAllocator {
  std::vector<char> _data;
  void *data;
  void *_cuda_data{};
  std::size_t size;
  bool gpu;

  void *head;
  void *tail;

 public:
  UnifiedAllocator() {
    data = nullptr;
  }

  UnifiedAllocator(std::size_t size, bool gpu);

#if defined(TC_GPU)
  __device__ __host__ void *alloc(int size) {
    void *ret;
    while (true) {
      auto old_head = head;
      auto new_head =
          (char *)atomicCAS(&(unsigned long)head, old_head, old_head + size);
      if (new_head == (char *)old_head + size) {
        ret = old_head;
        break;
      }
    }
  }
#else
  std::mutex lock;
  void *alloc(int size) {
    printf("this %p\n", this);
    std::lock_guard<std::mutex> _(lock);
    auto ret = head;
    head = (char *)head + size;
    return ret;
  }
#endif

  ~UnifiedAllocator();

  void memset(unsigned char val);

  bool initialized() const {
    return data != nullptr;
  }

  UnifiedAllocator operator=(const UnifiedAllocator &) = delete;

  /*
  UnifiedAllocator(UnifiedAllocator &&o) noexcept {
    (*this) = std::move(o);
  }

  UnifiedAllocator &operator=(UnifiedAllocator &&o) noexcept {
    std::swap(_data, o._data);
    data = o.data;
    o.data = nullptr;
    device = o.device;
    size = o.size;
    _cuda_data = o._cuda_data;
    return *this;
  }
  */

  static void create();

  static void free();
};

}  // namespace Tlang
}  // namespace taichi
