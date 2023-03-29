#pragma once
#include <mutex>
#include <vector>
#include <memory>
#include <map>

#include "taichi/rhi/arch.h"
#include "taichi/rhi/device.h"

namespace taichi::lang {

class MemoryPool;

// This class can only be accessed by MemoryPool
class UnifiedAllocator {
 private:
  struct MemoryChunk {
    bool is_exclusive;
    uint8 *data;
    uint8 *head;
    uint8 *tail;
  };

  static const std::size_t default_allocator_size;

  UnifiedAllocator(Arch arch);

  void *allocate(std::size_t size,
                 std::size_t alignment,
                 bool exclusive = false);

  void release(size_t sz, uint64_t *ptr);

  Arch arch_;
  std::map<uint8 *, MemoryChunk> chunks;

  friend class MemoryPool;
};

}  // namespace taichi::lang
