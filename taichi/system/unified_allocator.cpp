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
  MemoryChunk &current_chunk = chunks.rbegin()->second;

  uint8 *head = current_chunk.head;
  uint8 *tail = current_chunk.tail;
  uint8 *data = current_chunk.data;

  auto ret =
      head + alignment - 1 - ((std::size_t)head + alignment - 1) % alignment;
  TI_TRACE("UM [data={}] allocate() request={} remain={}", (intptr_t)data, size,
           (tail - head));
  head = ret + allocation_size;

  if (!exclusive) {
    // Do not allocate large memory chunks for "exclusive" allocation
    // to increate memory & performance efficiency
    allocation_size = std::max(allocation_size, default_allocator_size);
  }

  if (head > tail || exclusive) {
    // allocation failed, start with a new chunk
    MemoryChunk chunk;

    TI_TRACE("Allocating virtual address space of size {} MB",
             allocation_size / 1024 / 1024);

    void *ptr =
        MemoryPool::get_instance(arch_).allocate_raw_memory(allocation_size);
    chunk.data = (uint8 *)ptr;
    chunk.head = chunk.data;
    chunk.tail = chunk.head + allocation_size;
    chunk.is_exclusive = exclusive;

    chunks[chunk.data] = std::move(chunk);

    TI_ASSERT(chunk.data != nullptr);
    TI_ASSERT(uint64(chunk.data) % MemoryPool::page_size == 0);

    return chunk.data;
  } else {
    // success
    TI_ASSERT((std::size_t)ret % alignment == 0);
    current_chunk.head = head;
    return ret;
  }
}

void UnifiedAllocator::release(size_t sz, uint64_t *ptr) {
  // UnifiedAllocator is special in that it never reuses the previously
  // allocated memory We have to release the entire memory chunk to avoid memory
  // leak
  if (!chunks.count((uint8 *)ptr)) {
    TI_ERROR("UnifiedAllocator::release(): invalid pointer {}", (uint64_t)ptr);
  }

  auto &chunk = chunks[(uint8 *)ptr];
  TI_ASSERT(chunk.data == (uint8 *)ptr);
  TI_ASSERT(chunk.is_exclusive);

  chunks.erase((uint8 *)ptr);

  // MemoryPool is responsible for releasing the raw memory
}

}  // namespace taichi::lang
