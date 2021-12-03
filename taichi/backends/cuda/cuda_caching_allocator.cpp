#include "taichi/backends/cuda/cuda_caching_allocator.h"

namespace taichi {
namespace lang {
namespace cuda {

CudaCachingAllocator::CudaCachingAllocator(Device *device) : device_(device) {
}

uint64_t *CudaCachingAllocator::allocate(
    const Device::LlvmRuntimeAllocParams &params) {
  uint64_t *ret{nullptr};
  auto size_aligned = taichi::iroundup(params.size, taichi_page_size);
  auto it_blk = mem_blocks_.lower_bound(size_aligned);

  if (it_blk != mem_blocks_.end()) {
    size_t remaining_sz = it_blk->first - size_aligned;
    if (remaining_sz > 0) {
      TI_ASSERT(remaining_sz % taichi_page_size == 0);
      auto remaining_head =
          reinterpret_cast<uint8_t *>(it_blk->second) + size_aligned;
      mem_blocks_.insert(
          {remaining_sz, reinterpret_cast<uint64_t *>(remaining_head)});
    }
    ret = it_blk->second;
    mem_blocks_.erase(it_blk);
  } else {
    ret = device_->allocate_llvm_runtime_memory_jit(params);
  }
  return ret;
}

void CudaCachingAllocator::release(size_t sz, uint64_t *ptr) {
  mem_blocks_.insert({sz, ptr});
}

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
