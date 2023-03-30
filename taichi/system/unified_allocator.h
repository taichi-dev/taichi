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
    void *data;
    void *head;
    void *tail;
  };

  static const std::size_t default_allocator_size;

  UnifiedAllocator(Arch arch);

  void *allocate(std::size_t size,
                 std::size_t alignment,
                 bool exclusive = false);

  bool is_releaseable(void *ptr);
  void release(size_t sz, void *ptr);

  Arch arch_;
  std::vector<MemoryChunk> chunks_;

  // For fast lookup of the chunk index wrt an allocated pointer upon release
  std::unordered_map<void *, size_t> ptr_chunk_idx_map_;

  friend class MemoryPool;
};

}  // namespace taichi::lang
