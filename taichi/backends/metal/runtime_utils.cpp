#include "taichi/backends/metal/runtime_utils.h"

#include <cstring>

#include "taichi/inc/constants.h"
#include "taichi/math/arithmetic.h"
#include "taichi/system/memory_pool.h"

namespace taichi {
namespace lang {
namespace metal {

BufferMemoryView::BufferMemoryView(std::size_t size, MemoryPool *mem_pool) {
  // Both |ptr_| and |size_| must be aligned to page size.
  size_ = iroundup(size, taichi_page_size);
  ptr_ = (char *)mem_pool->allocate(size_, /*alignment=*/taichi_page_size);
  TI_ASSERT(ptr_ != nullptr);
  std::memset(ptr_, 0, size_);
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
