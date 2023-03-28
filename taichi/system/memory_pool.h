#pragma once
#include "taichi/common/core.h"
#include "taichi/system/unified_allocator.h"
#define TI_RUNTIME_HOST
#include "taichi/runtime/llvm/runtime_module/mem_request.h"
#undef TI_RUNTIME_HOST
#include "taichi/rhi/device.h"
#include <mutex>
#include <vector>
#include <memory>
#include <thread>

namespace taichi::lang {

class UnifiedAllocator;

// A memory pool that runs on the host

class TI_DLL_EXPORT MemoryPool {
 public:
  static MemoryPool &get_instance(Arch arch);

  std::vector<std::unique_ptr<UnifiedAllocator>> allocators;
  static constexpr std::size_t default_allocator_size =
      1 << 30;                                    // 1 GB per allocator
  static constexpr size_t page_size = (1 << 12);  // 4 KB page size by default
  std::mutex mut_allocators;
  std::mutex mut_raw_alloc;

  void *allocate(std::size_t size, std::size_t alignment);
  void release(std::size_t size, void *ptr);

  ~MemoryPool();
  MemoryPool(Arch arch);

 private:
  void *allocate_raw_memory(std::size_t size);

  // Only MemoryPool can deallocate raw memory
  void deallocate_raw_memory(void *ptr);

  // All the raw memory allocated from OS/Driver
  // We need to keep track of them to guarantee that they are freed
  std::map<void *, std::size_t> raw_memory_chunks_;

  Arch arch_;

  friend class UnifiedAllocator;
};

}  // namespace taichi::lang
