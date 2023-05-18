#include "taichi/rhi/llvm/allocator.h"
#include "taichi/runtime/llvm/snode_tree_buffer_manager.h"

namespace taichi::lang {

CachingAllocator::CachingAllocator(bool merge_upon_release)
    : merge_upon_release_(merge_upon_release) {
}

void CachingAllocator::merge_and_insert(uint8_t *ptr, std::size_t size) {
  // merge with right block
  if (ptr_map_[ptr + size]) {
    std::size_t tmp = ptr_map_[ptr + size];
    mem_blocks_.erase(std::make_pair(tmp, ptr + size));
    ptr_map_.erase(ptr + size);
    size += tmp;
  }
  // merge with left block
  auto map_it = ptr_map_.lower_bound(ptr);
  if (map_it != ptr_map_.begin()) {
    auto x = *--map_it;
    if (x.first + x.second == ptr) {
      mem_blocks_.erase(std::make_pair(x.second, x.first));
      ptr_map_.erase(x.first);
      ptr = x.first;
      size += x.second;
    }
  }
  mem_blocks_.insert(std::make_pair(size, ptr));
  ptr_map_[ptr] = size;
}

uint64_t *CachingAllocator::allocate(
    LlvmDevice *device,
    const LlvmDevice::LlvmRuntimeAllocParams &params) {
  uint64_t *ret{nullptr};
  auto size_aligned = taichi::iroundup(params.size, taichi_page_size);
  auto it_blk = mem_blocks_.lower_bound(std::make_pair(size_aligned, nullptr));

  if (it_blk != mem_blocks_.end()) {
    size_t remaining_sz = it_blk->first - size_aligned;
    if (remaining_sz > 0) {
      TI_ASSERT(remaining_sz % taichi_page_size == 0);
      auto remaining_head =
          reinterpret_cast<uint8_t *>(it_blk->second) + size_aligned;
      mem_blocks_.insert(std::make_pair(remaining_sz, remaining_head));
      ptr_map_.insert(std::make_pair(remaining_head, remaining_sz));
    }
    ret = reinterpret_cast<uint64_t *>(it_blk->second);
    mem_blocks_.erase(it_blk);
    ptr_map_.erase(it_blk->second);

  } else {
    ret = reinterpret_cast<uint64_t *>(
        device->allocate_llvm_runtime_memory_jit(params));
  }
  return ret;
}

void CachingAllocator::release(size_t sz, uint64_t *ptr) {
  if (merge_upon_release_) {
    merge_and_insert(reinterpret_cast<uint8_t *>(ptr), sz);
  } else {
    if (sz >= taichi_page_size) {
      mem_blocks_.insert(std::make_pair(sz, reinterpret_cast<uint8_t *>(ptr)));
    }
  }
}

}  // namespace taichi::lang
