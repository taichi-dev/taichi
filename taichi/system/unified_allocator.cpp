// Virtual memory allocator for CPU/GPU

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"

#endif
#include "taichi/util/lang_util.h"
#include "taichi/system/unified_allocator.h"
#include "taichi/system/memory_pool.h"
#include "taichi/system/timer.h"
#include "taichi/rhi/cpu/cpu_device.h"
#include <string>

namespace taichi::lang {

const std::size_t UnifiedAllocator::default_allocator_size =
    1 << 30;  // 1 GB per allocator

static void remove_memory_chunk(std::vector<UnifiedAllocator::MemoryChunk> &vec,
                                size_t idx) {
  if (idx + 1 < vec.size()) {
    std::swap(vec[idx], vec.back());
  }

  vec.pop_back();

  // swap it back so it does not influence the last memory chunk to reuse
  if (idx + 1 < vec.size()) {
    std::swap(vec[idx], vec.back());
  }
}

UnifiedAllocator::UnifiedAllocator(Arch arch) : arch_(arch) {
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
    // Try reusing the last chunk
    MemoryChunk &current_chunk = chunks_.back();

    auto head = (std::size_t)current_chunk.head;
    auto tail = (std::size_t)current_chunk.tail;
    auto data = (std::size_t)current_chunk.data;

    auto ret = head + alignment - 1 - (head + alignment - 1) % alignment;
    TI_TRACE("UM [data={}] allocate() request={} remain={}", (intptr_t)data,
             size, (tail - head));
    head = ret + allocation_size;

    if (head <= tail) {
      // success
      TI_ASSERT(ret % alignment == 0);
      current_chunk.head = (void *)head;
      return (void *)ret;
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
      MemoryPool::get_instance(arch_).allocate_raw_memory(allocation_size);
  chunk.data = ptr;
  chunk.head = chunk.data;
  chunk.tail = (void *)((std::size_t)chunk.head + allocation_size);
  chunk.is_exclusive = exclusive;

  TI_ASSERT(chunk.data != nullptr);
  TI_ASSERT(uint64(chunk.data) % MemoryPool::page_size == 0);

  chunks_.emplace_back(std::move(chunk));
  return ptr;
}

bool UnifiedAllocator::release(size_t sz, void *ptr) {
  // UnifiedAllocator is special in that it never reuses the previously
  // allocated memory We have to release the entire memory chunk to avoid memory
  // leak
  for (size_t chunk_idx = 0; chunk_idx < chunks_.size(); chunk_idx++) {
    auto &chunk = chunks_[chunk_idx];

    if (chunk.data == ptr) {
      TI_ASSERT(chunk.is_exclusive);
      remove_memory_chunk(chunks_, chunk_idx);

      // MemoryPool is responsible for releasing the raw memory
      return true;
    }
  }

  return false;
}

}  // namespace taichi::lang
