#pragma once

#include <taichi/common/util.h>
#include "common.h"
#include <mutex>
#include <vector>
#include <memory>
#include <thread>
#include "unified_allocator.h"
#include "legacy_kernel.h"

TLANG_NAMESPACE_BEGIN

class Program;

// A memory pool that runs on the host

class MemoryPool {
 public:
  std::vector<std::unique_ptr<UnifiedAllocator>> allocators;
  static constexpr std::size_t default_allocator_size =
      1 << 30;  // 1 GB per allocator
  bool terminating, killed;
  std::mutex mut;
  std::unique_ptr<std::thread> th;
  int processed_tail;
  Program *prog;

  MemRequestQueue *queue;

  MemoryPool(Program *prog);

  void *allocate(std::size_t size, std::size_t alignment);

  void set_queue(MemRequestQueue *queue);

  void daemon();

  void terminate();

  ~MemoryPool();
};

TLANG_NAMESPACE_END
