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

// A memory pool that runs on the host

class TI_DLL_EXPORT MemoryPool {
 public:
  std::vector<std::unique_ptr<UnifiedAllocator>> allocators;
  static constexpr std::size_t default_allocator_size =
      1 << 30;  // 1 GB per allocator
  std::mutex mut_allocators;

  void *cuda_stream{nullptr};
  void *amdgpu_stream{nullptr};

  // In the future we wish to move the MemoryPool inside each Device
  // so that the memory allocated from each Device can be used as-is.
  MemoryPool(Arch arch, Device *device);

  void *allocate(std::size_t size, std::size_t alignment);

  ~MemoryPool();

 private:
  static constexpr bool use_cuda_stream = false;
  static constexpr bool use_amdgpu_stream = false;
  Arch arch_;
  Device *device_;
};

}  // namespace taichi::lang
