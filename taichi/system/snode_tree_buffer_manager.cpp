#include "snode_tree_buffer_manager.h"
#include "taichi/program/program.h"
#ifdef TI_WITH_LLVM
#include "taichi/llvm/llvm_program.h"
#endif

TLANG_NAMESPACE_BEGIN

SNodeTreeBufferManager::SNodeTreeBufferManager(ProgramImpl *prog)
    : prog_(prog) {
  TI_TRACE("SNode tree buffer manager created.");
}

void SNodeTreeBufferManager::merge_and_insert(Ptr ptr, std::size_t size) {
  // merge with right block
  if (ptr_map_[ptr + size]) {
    std::size_t tmp = ptr_map_[ptr + size];
    size_set_.erase(std::make_pair(tmp, ptr + size));
    ptr_map_.erase(ptr + size);
    size += tmp;
  }
  // merge with left block
  auto map_it = ptr_map_.lower_bound(ptr);
  if (map_it != ptr_map_.begin()) {
    auto x = *--map_it;
    if (x.first + x.second == ptr) {
      size_set_.erase(std::make_pair(x.second, x.first));
      ptr_map_.erase(x.first);
      ptr = x.first;
      size += x.second;
    }
  }
  size_set_.insert(std::make_pair(size, ptr));
  ptr_map_[ptr] = size;
}

Ptr SNodeTreeBufferManager::allocate(JITModule *runtime_jit,
                                     void *runtime,
                                     std::size_t size,
                                     std::size_t alignment,
                                     const int snode_tree_id,
                                     uint64 *result_buffer) {
#ifdef TI_WITH_LLVM
  TI_TRACE("allocating memory for SNode Tree {}", snode_tree_id);
  TI_ASSERT_INFO(snode_tree_id < kMaxNumSnodeTreesLlvm,
                 "LLVM backend supports up to {} snode trees",
                 kMaxNumSnodeTreesLlvm);
  auto set_it = size_set_.lower_bound(std::make_pair(size, nullptr));
  if (set_it == size_set_.end()) {
    runtime_jit->call<void *, std::size_t, std::size_t>(
        "runtime_memory_allocate_aligned", runtime, size, alignment);
    LlvmProgramImpl *llvm_prog = static_cast<LlvmProgramImpl *>(prog_);
    auto ptr = llvm_prog->fetch_result<Ptr>(
        taichi_result_buffer_runtime_query_id, result_buffer);
    roots_[snode_tree_id] = ptr;
    sizes_[snode_tree_id] = size;
    return ptr;
  } else {
    auto x = *set_it;
    size_set_.erase(x);
    ptr_map_.erase(x.second);
    if (x.first - size > 0) {
      size_set_.insert(std::make_pair(x.first - size, x.second + size));
      ptr_map_[x.second + size] = x.first - size;
    }
    TI_ASSERT(x.second);
    roots_[snode_tree_id] = x.second;
    sizes_[snode_tree_id] = size;
    return x.second;
  }
#else
  TI_ERROR("Llvm disabled");
#endif
}

void SNodeTreeBufferManager::destroy(SNodeTree *snode_tree) {
  int snode_tree_id = snode_tree->id();
  TI_TRACE("Destroying SNode tree {}.", snode_tree_id);
  std::size_t size = sizes_[snode_tree_id];
  if (size == 0) {
    TI_DEBUG("SNode tree {} destroy failed.", snode_tree_id);
    return;
  }
  Ptr ptr = roots_[snode_tree_id];
  merge_and_insert(ptr, size);
  TI_DEBUG("SNode tree {} destroyed.", snode_tree_id);
}

TLANG_NAMESPACE_END
