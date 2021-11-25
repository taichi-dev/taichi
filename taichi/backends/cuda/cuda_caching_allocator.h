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

  void release(size_t sz, uint64_t *ptr);

  uint64_t *allocate(Device::AllocParamsLlvmRuntime &params);

  bool find_block(size_t sz) const;

 private:
  std::multimap<size_t, uint64_t *> mem_blocks_;
  Device *device_{nullptr};
};

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
