#pragma once
#include "taichi/common/core.h"
#include "taichi/rhi/common/unified_allocator.h"
#include "taichi/rhi/device.h"
#include <mutex>
#include <vector>
#include <memory>
#include <thread>

namespace taichi::lang {

// A memory pool that runs on the host

class TI_DLL_EXPORT DeviceMemoryPool {
 public:
  static const size_t page_size;

  static DeviceMemoryPool &get_instance();

  void *allocate(std::size_t size, std::size_t alignment, bool managed = false);
  void release(std::size_t size, void *ptr);
  void reset();
  DeviceMemoryPool();
  ~DeviceMemoryPool();

 protected:
  void *allocate_raw_memory(std::size_t size, bool managed = false);
  void deallocate_raw_memory(void *ptr);

  // All the raw memory allocated from OS/Driver
  // We need to keep track of them to guarantee that they are freed
  std::map<void *, std::size_t> raw_memory_chunks_;

  std::mutex mut_allocation_;
};

}  // namespace taichi::lang
