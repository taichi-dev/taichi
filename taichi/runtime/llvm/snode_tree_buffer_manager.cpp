#include "snode_tree_buffer_manager.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"

namespace taichi::lang {

SNodeTreeBufferManager::SNodeTreeBufferManager(
    LlvmRuntimeExecutor *runtime_exec)
    : runtime_exec_(runtime_exec) {
  TI_TRACE("SNode tree buffer manager created.");
}

Ptr SNodeTreeBufferManager::allocate(std::size_t size,
                                     const int snode_tree_id,
                                     uint64 *result_buffer) {
  auto devalloc = runtime_exec_->allocate_memory_on_device(size, result_buffer);
  snode_tree_id_to_device_alloc_[snode_tree_id] = devalloc;
  return (Ptr)runtime_exec_->get_device_alloc_info_ptr(devalloc);
}

void SNodeTreeBufferManager::destroy(SNodeTree *snode_tree) {
  auto devalloc = snode_tree_id_to_device_alloc_[snode_tree->id()];
  runtime_exec_->deallocate_memory_on_device(devalloc);
  snode_tree_id_to_device_alloc_.erase(snode_tree->id());
}

}  // namespace taichi::lang
