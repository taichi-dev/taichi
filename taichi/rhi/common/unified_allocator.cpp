// Virtual memory allocator for CPU/GPU

#include "taichi/rhi/common/unified_allocator.h"
#include "taichi/rhi/common/host_memory_pool.h"
#include <string>

namespace taichi::lang {

const std::size_t UnifiedAllocator::default_allocator_size =
    1 << 30;  // 1 GB per allocator

template <typename T>
static void swap_erase_vector(std::vector<T> &vec, size_t idx) {
  bool is_last = idx == vec.size() - 1;
  TI_ASSERT(idx < vec.size());

  if (!is_last) {
    std::swap(vec[idx], vec.back());
  }

  vec.pop_back();

  // There's no need to swap back since we'll iterate the memory chunks to
  // search for reusable memory
}

UnifiedAllocator::UnifiedAllocator() {
}

void *UnifiedAllocator::allocate(std::size_t size,
                                 std::size_t alignment,
                                 bool exclusive) {
  // UnifiedAllocator never reuses the previously allocated memory
  // just move the head forward util depleting all the free memory

  // Note: put mutex on MemoryPool instead of Allocator, since Allocators are
  // transparent to user code
  std::size_t allocation_size = size;
  if (!chunks_.empty() && !exclusive) {
    // Search for a non-exclusive chunk that has enough space
    for (size_t chunk_id = 0; chunk_id < chunks_.size(); chunk_id++) {
      auto &chunk = chunks_[chunk_id];
      if (chunk.is_exclusive) {
        continue;
      }
      auto head = (std::size_t)chunk.head;
      auto tail = (std::size_t)chunk.tail;
      auto data = (std::size_t)chunk.data;
      auto ret = head + alignment - 1 - (head + alignment - 1) % alignment;
      TI_TRACE("UM [data={}] allocate() request={} remain={}", (intptr_t)data,
               size, (tail - head));
      head = ret + allocation_size;
      if (head <= tail) {
        // success
        TI_ASSERT(ret % alignment == 0);
        chunk.head = (void *)head;
        return (void *)ret;
      }
    }
  }

  // Allocate a new chunk
  MemoryChunk chunk;

  if (!exclusive) {
    // Do not allocate large memory chunks for "exclusive" allocation
    // to increate memory & performance efficiency
    allocation_size = std::max(allocation_size, default_allocator_size);
  }

  TI_TRACE("Allocating virtual address space of size {} MB",
           allocation_size / 1024 / 1024);

  void *ptr =
      HostMemoryPool::get_instance().allocate_raw_memory(allocation_size);
  chunk.data = ptr;
  chunk.head = chunk.data;
  chunk.tail = (void *)((std::size_t)chunk.head + allocation_size);
  chunk.is_exclusive = exclusive;

  TI_ASSERT(chunk.data != nullptr);
  TI_ASSERT(uint64(chunk.data) % HostMemoryPool::page_size == 0);

  chunks_.emplace_back(std::move(chunk));
  return ptr;
}

bool UnifiedAllocator::release(size_t sz, void *ptr) {
  // UnifiedAllocator is special in that it never reuses the previously
  // allocated memory We have to release the entire memory chunk to avoid memory
  // leak
  int remove_idx = -1;
  for (size_t chunk_idx = 0; chunk_idx < chunks_.size(); chunk_idx++) {
    auto &chunk = chunks_[chunk_idx];

    if (chunk.data == ptr) {
      TI_ASSERT(chunk.is_exclusive);
      remove_idx = chunk_idx;
    }
  }

  if (remove_idx != -1) {
    swap_erase_vector<MemoryChunk>(chunks_, remove_idx);
    // MemoryPool is responsible for releasing the raw memory
    return true;
  }

  return false;
}

}  // namespace taichi::lang
