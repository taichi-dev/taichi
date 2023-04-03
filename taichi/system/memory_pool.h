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
  static const size_t page_size;

  static MemoryPool &get_instance(Arch arch);

  void *allocate(std::size_t size,
                 std::size_t alignment,
                 bool releasable = false);
  void release(std::size_t size, void *ptr);
  void reset();

  ~MemoryPool();

 private:
  MemoryPool(Arch arch);

  void *allocate_raw_memory(std::size_t size);
  void deallocate_raw_memory(void *ptr);

  // All the raw memory allocated from OS/Driver
  // We need to keep track of them to guarantee that they are freed
  std::map<void *, std::size_t> raw_memory_chunks_;
  std::unique_ptr<UnifiedAllocator> allocator_;
  std::mutex mut_allocation_;
  Arch arch_;

  friend class UnifiedAllocator;
};

}  // namespace taichi::lang
