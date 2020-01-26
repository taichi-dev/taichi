#pragma once

#include "common.h"
#include <mutex>
#include <vector>
#include <memory>
#include <thread>
#include "unified_allocator.h"

TLANG_NAMESPACE_BEGIN

// A memory pool that runs on the host

class MemoryPool {
 public:
  std::vector<std::unique_ptr<UnifiedAllocator>> allocators;
  static constexpr std::size_t allocator_size = 1 << 30;  // 1 GB per allocator
  bool killed;
  std::mutex mut;
  std::unique_ptr<std::thread> th;

  MemoryPool();

  void daemon();

  ~MemoryPool();
};

TLANG_NAMESPACE_END
