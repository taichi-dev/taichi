#pragma once

#include <cstddef>

namespace taichi {
namespace lang {

class MemoryPool;

namespace metal {

// This class requests the Metal buffer memory of |size| bytes from |mem_pool|.
// Once allocated, it does not own the memory (hence the name "view"). Instead,
// GC is deferred to the memory pool.
class BufferMemoryView {
 public:
  BufferMemoryView(std::size_t size, MemoryPool *mem_pool);
  // Move only
  BufferMemoryView(BufferMemoryView &&) = default;
  BufferMemoryView &operator=(BufferMemoryView &&) = default;
  BufferMemoryView(const BufferMemoryView &) = delete;
  BufferMemoryView &operator=(const BufferMemoryView &) = delete;

  inline size_t size() const {
    return size_;
  }
  inline char *ptr() const {
    return ptr_;
  }

 private:
  std::size_t size_{0};
  char *ptr_{nullptr};
};

}  // namespace metal
}  // namespace lang
}  // namespace taichi
