#pragma once

#include "taichi/backends/device.h"
#include "taichi/common/core.h"
#include <stdint.h>
#include <map>

namespace taichi {
namespace lang {
namespace cuda {

class CudaCachingAllocator {
 public:
  CudaCachingAllocator(Device *device);

  uint64_t *allocate(const Device::LlvmRuntimeAllocParams &params);
  void release(size_t sz, uint64_t *ptr);

 private:
  std::multimap<size_t, uint64_t *> mem_blocks_;
  Device *device_{nullptr};
  bool find_block(size_t sz) const;
};

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
