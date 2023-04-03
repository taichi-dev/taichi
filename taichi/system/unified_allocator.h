#pragma once
#include <mutex>
#include <vector>
#include <memory>
#include <unordered_set>

#include "taichi/rhi/arch.h"
#include "taichi/rhi/device.h"

namespace taichi::lang {

class MemoryPool;

// This class can only have one instance
class UnifiedAllocator {
  std::size_t size_;
  Arch arch_;

  // put these two on the unified memory so that GPU can have access
 private:
  bool is_exclusive;
  uint8 *data;
  uint8 *head;
  uint8 *tail;

 public:
  UnifiedAllocator(std::size_t size, Arch arch, bool is_exclusive = false);

  ~UnifiedAllocator();

  void *allocate(std::size_t size, std::size_t alignment);

  void release(size_t sz, uint64_t *ptr);
  bool is_releasable(uint64_t *ptr) const;

  void memset(unsigned char val);

  bool initialized() const {
    return data != nullptr;
  }

  UnifiedAllocator operator=(const UnifiedAllocator &) = delete;
};

}  // namespace taichi::lang
