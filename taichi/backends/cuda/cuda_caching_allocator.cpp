#include "taichi/backends/cuda/cuda_caching_allocator.h"

namespace taichi {
namespace lang {
namespace cuda {

CudaCachingAllocator::CudaCachingAllocator(Device *device) : device_(device) {
}

uint64_t *CudaCachingAllocator::allocate(
    const Device::LlvmRuntimeAllocParams &params) {
  uint64_t *ret{nullptr};
  if (find_block(params.size)) {
    auto blk = mem_blocks_.find(params.size);
    ret = blk->second;
    mem_blocks_.erase(blk);
  } else {
    ret = device_->allocate_llvm_runtime_memory_jit(params);
  }
  return ret;
}

void CudaCachingAllocator::release(size_t sz, uint64_t *ptr) {
  // mem_blocks_.insert(std::pair<size_t, uint64_t *>(sz, ptr));
  mem_blocks_.insert({sz, ptr});
}

bool CudaCachingAllocator::find_block(size_t sz) const {
  return mem_blocks_.find(sz) != mem_blocks_.end();
}

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
