#pragma once
#include "taichi/common/core.h"
#include "taichi/rhi/device.h"
#include "taichi/rhi/llvm/llvm_device.h"
#include "taichi/rhi/llvm/allocator.h"
#include <mutex>
#include <vector>
#include <memory>
#include <thread>

namespace taichi::lang {

// A memory pool that runs on the host

class TI_DLL_EXPORT DeviceMemoryPool {
 public:
  std::unique_ptr<CachingAllocator> allocator_{nullptr};
  static const size_t page_size;

  static DeviceMemoryPool &get_instance(bool merge_upon_release = true);

  void *allocate_with_cache(LlvmDevice *device,
                            const LlvmDevice::LlvmRuntimeAllocParams &params);
  void *allocate(std::size_t size, std::size_t alignment, bool managed = false);
  void release(std::size_t size, void *ptr, bool release_raw = false);
  void reset();
  explicit DeviceMemoryPool(bool merge_upon_release);
  ~DeviceMemoryPool();

 protected:
  void *allocate_raw_memory(std::size_t size, bool managed = false);
  void deallocate_raw_memory(void *ptr);

  // All the raw memory allocated from OS/Driver
  // We need to keep track of them to guarantee that they are freed
  std::map<void *, std::size_t> raw_memory_chunks_;

  std::mutex mut_allocation_;
  bool merge_upon_release_ = true;
};

}  // namespace taichi::lang
