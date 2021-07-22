#include "snode_tree_buffer_manager.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

SNodeTreeBufferManager::SNodeTreeBufferManager(Program *prog) : prog(prog) {
  TI_TRACE("SNode tree buffer manager created.");
}

void *SNodeTreeBufferManager::allocate(std::size_t size,
                                       std::size_t alignment,
                                       const int snode_tree_id) {
  void *ret = nullptr;
  if (size >= prog->config.memory_allocate_critical_size) {
    // allocate a seperate memory for large SNode tree
    allocators[snode_tree_id] =
        std::make_unique<UnifiedAllocator>(size, prog->config.arch);
    ret = allocators[snode_tree_id]->allocate(size, alignment);
  } else {
    ret = prog->memory_pool->allocate(size, alignment);
  }
  TI_ASSERT(ret);
  return ret;
}

void SNodeTreeBufferManager::destroy(const int snode_tree_id) {
  TI_TRACE("Destroying SNode tree {}.", snode_tree_id);
  if (allocators.find(snode_tree_id) != allocators.end()) {
    allocators[snode_tree_id].reset();
  } else {
    TI_DEBUG("SNode tree {} destroy failed.", snode_tree_id);
  }
}

TLANG_NAMESPACE_END
