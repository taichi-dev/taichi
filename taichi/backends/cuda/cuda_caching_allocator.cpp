#include "taichi/backends/cuda/cuda_caching_allocator.h"

namespace taichi {
namespace lang {
namespace cuda {

bool CudaCachingAllocator::find_block(size_t sz) const {
  return mem_blocks.find(sz) != mem_blocks.end();
}

uint64_t *CudaCachingAllocator::allocate(size_t sz) {
  auto blk = mem_blocks.find(sz);
  uint64_t *ret = blk->second;
  mem_blocks.erase(blk);
  return ret;
}

void CudaCachingAllocator::release(size_t sz, uint64_t *ptr) {
  mem_blocks.insert(std::pair<size_t, uint64_t *>(sz, ptr));
}

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
