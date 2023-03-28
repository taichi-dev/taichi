#pragma once
#include <mutex>
#include <vector>
#include <memory>

#include "taichi/rhi/arch.h"
#include "taichi/rhi/device.h"

namespace taichi::lang {

class MemoryPool;

// This class can only have one instance
class UnifiedAllocator {
  std::size_t size_;

  // put these two on the unified memory so that GPU can have access
 public:
  uint8 *data;
  uint8 *head;
  uint8 *tail;

 public:
  UnifiedAllocator(std::size_t size, MemoryPool *memory_pool);

  ~UnifiedAllocator();

  void *allocate(std::size_t size, std::size_t alignment);

  void release(size_t sz, uint64_t *ptr);

  void memset(unsigned char val);

  bool initialized() const {
    return data != nullptr;
  }

  UnifiedAllocator operator=(const UnifiedAllocator &) = delete;
};

}  // namespace taichi::lang
