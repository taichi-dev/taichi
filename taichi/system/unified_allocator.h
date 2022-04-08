#pragma once
#include <mutex>
#include <vector>
#include <memory>

#include "taichi/backends/arch.h"
#include "taichi/backends/device.h"

namespace taichi {
class VirtualMemoryAllocator;
}

TLANG_NAMESPACE_BEGIN

// This class can only have one instance
class UnifiedAllocator {
  std::unique_ptr<VirtualMemoryAllocator> cpu_vm_;
  std::size_t size_;
  Arch arch_;

  // put these two on the unified memory so that GPU can have access
 public:
  uint8 *data;
  DeviceAllocation alloc{kDeviceNullAllocation};
  uint8 *head;
  uint8 *tail;
  std::mutex lock;

 public:
  UnifiedAllocator(std::size_t size, Arch arch, Device *device);

  ~UnifiedAllocator();

  void *allocate(std::size_t size, std::size_t alignment) {
    std::lock_guard<std::mutex> _(lock);
    auto ret =
        head + alignment - 1 - ((std::size_t)head + alignment - 1) % alignment;
    TI_TRACE("UM [data={}] allocate() request={} remain={}", (intptr_t)data,
             size, (tail - head));
    head = ret + size;
    if (head > tail) {
      // allocation failed
      return nullptr;
    } else {
      // success
      TI_ASSERT((std::size_t)ret % alignment == 0);
      return ret;
    }
  }

  void memset(unsigned char val);

  bool initialized() const {
    return data != nullptr;
  }

  UnifiedAllocator operator=(const UnifiedAllocator &) = delete;

 private:
  Device *device_{nullptr};
};

TLANG_NAMESPACE_END
