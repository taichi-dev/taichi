#pragma once

#include "taichi/common/core.h"
#include "taichi/math/arithmetic.h"
#include "taichi/rhi/llvm/llvm_device.h"
#include <stdint.h>
#include <map>

namespace taichi {
namespace lang {
namespace amdgpu {

class AmdgpuCachingAllocator {
 public:
  AmdgpuCachingAllocator(LlvmDevice *device);

  uint64_t *allocate(const LlvmDevice::LlvmRuntimeAllocParams &params);
  void release(size_t sz, uint64_t *ptr);

 private:
  std::multimap<size_t, uint64_t *> mem_blocks_;
  LlvmDevice *device_{nullptr};
};

}  // namespace amdgpu
}  // namespace lang
}  // namespace taichi
