#pragma once

#include "taichi/common/core.h"
#include "taichi/math/arithmetic.h"
#include "taichi/llvm/llvm_device.h"
#include <stdint.h>
#include <map>

namespace taichi {
namespace lang {
namespace cuda {

class CudaCachingAllocator {
 public:
  CudaCachingAllocator(LlvmDevice *device);

  uint64_t *allocate(const LlvmDevice::LlvmRuntimeAllocParams &params);
  void release(size_t sz, uint64_t *ptr);

 private:
  std::multimap<size_t, uint64_t *> mem_blocks_;
  LlvmDevice *device_{nullptr};
};

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
