#pragma once
#include "taichi/common/core.h"
#include "taichi/system/unified_allocator.h"
#define TI_RUNTIME_HOST
#include "taichi/runtime/llvm/mem_request.h"

#include <mutex>
#include <vector>
#include <memory>
#include <thread>

TLANG_NAMESPACE_BEGIN

class Program;

// A memory pool that runs on the host

class MemoryPool {
 public:
  std::vector<std::unique_ptr<UnifiedAllocator>> allocators;
  std::vector<int> allocator_snode_tree_id;
  std::vector<std::unique_ptr<UnifiedAllocator>> snode_tree_allocators;
  static constexpr std::size_t default_allocator_size =
      1 << 30;  // 1 GB per allocator
  bool terminating, killed;
  std::mutex mut;
  std::mutex mut_allocators;
  std::unique_ptr<std::thread> th;
  int processed_tail;
  bool use_unified_memory;
  Program *prog;

  MemRequestQueue *queue;
  void *cuda_stream{nullptr};

  MemoryPool(Program *prog);

  template <typename T>
  T fetch(volatile void *ptr);

  template <typename T>
  void push(volatile T *dest, const T &val);

  void *allocate(std::size_t size,
                 std::size_t alignment,
                 const int snode_tree_id = -1);

  void destroy_snode_tree(const int snode_tree_id);

  void set_queue(MemRequestQueue *queue);

  void daemon();

  void terminate();

  ~MemoryPool();

 private:
  static constexpr bool use_cuda_stream = false;
};

TLANG_NAMESPACE_END
