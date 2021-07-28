#include "snode_tree_buffer_manager.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

SNodeTreeBufferManager::SNodeTreeBufferManager(Program *prog) : prog(prog) {
  TI_TRACE("SNode tree buffer manager created.");
}

void SNodeTreeBufferManager::merge_and_insert(Ptr ptr, std::size_t size) {
  // merge with right block
  while (Ptr_map[ptr + size]) {
    std::size_t tmp = Ptr_map[ptr + size];
    size_set.erase(std::make_pair(tmp, ptr + size));
    Ptr_map.erase(ptr + size);
    size += tmp;
  }
  // merge with left block
  while (Ptr_map.lower_bound(ptr) != Ptr_map.begin()) {
    std::pair<Ptr, std::size_t> x = *--Ptr_map.lower_bound(ptr);
    if (x.first + x.second != ptr) {
      break;
    }
    size_set.erase(std::make_pair(x.second, x.first));
    Ptr_map.erase(x.first);
    ptr = x.first;
    size += x.second;
  }
  size_set.insert(std::make_pair(size, ptr));
  Ptr_map[ptr] = size;
}

Ptr SNodeTreeBufferManager::allocate(JITModule *runtime_jit,
                                     void *runtime,
                                     std::size_t size,
                                     std::size_t alignment,
                                     const int snode_tree_id) {
  TI_TRACE("allocating memory for SNode Tree {}", snode_tree_id);
  if (size_set.lower_bound(std::make_pair(size, nullptr)) == size_set.end()) {
    runtime_jit->call<void *, std::size_t, std::size_t>(
        "runtime_snode_tree_allocate_aligned", runtime, size, alignment);
    auto ptr = prog->fetch_result<Ptr>(taichi_result_buffer_runtime_query_id);
    roots[snode_tree_id] = ptr;
    sizes[snode_tree_id] = size;
    return ptr;
  } else {
    std::pair<std::size_t, Ptr> x =
        *size_set.lower_bound(std::make_pair(size, nullptr));
    size_set.erase(x);
    Ptr_map.erase(x.second);
    if (x.first - size > 0) {
      size_set.insert(std::make_pair(x.first - size, x.second + size));
      Ptr_map[x.second + size] = x.first - size;
    }
    TI_ASSERT(x.second);
    roots[snode_tree_id] = x.second;
    sizes[snode_tree_id] = size;
    return x.second;
  }
}

void SNodeTreeBufferManager::destroy(const int snode_tree_id) {
  TI_TRACE("Destroying SNode tree {}.", snode_tree_id);
  std::size_t size = sizes[snode_tree_id];
  if (size == 0) {
    TI_DEBUG("SNode tree {} destroy failed.", snode_tree_id);
    return;
  }
  Ptr ptr = roots[snode_tree_id];
  merge_and_insert(ptr, size);
  TI_DEBUG("SNode tree {} destroyed.", snode_tree_id);
}

TLANG_NAMESPACE_END
