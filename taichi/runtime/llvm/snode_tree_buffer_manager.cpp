#include "snode_tree_buffer_manager.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"

namespace taichi::lang {

SNodeTreeBufferManager::SNodeTreeBufferManager(
    LlvmRuntimeExecutor *runtime_exec)
    : runtime_exec_(runtime_exec) {
  TI_TRACE("SNode tree buffer manager created.");
}

DeviceAllocation SNodeTreeBufferManager::allocate(std::size_t size,
                                                  const int snode_tree_id,
                                                  uint64 *result_buffer) {
  std::cout << 1111 << std::endl;
  std::cout << size << std::endl;
  DeviceAllocation devalloc =
      runtime_exec_->allocate_memory_ndarray(size, result_buffer);
  std::cout << 2222 << std::endl;
  roots_[snode_tree_id] = devalloc;
  sizes_[snode_tree_id] = size;

  return devalloc;
}

void SNodeTreeBufferManager::destroy(SNodeTree *snode_tree) {
  int snode_tree_id = snode_tree->id();
  TI_TRACE("Destroying SNode tree {}.", snode_tree_id);
  std::size_t size = sizes_[snode_tree_id];
  if (size == 0) {
    TI_DEBUG("SNode tree {} destroy failed.", snode_tree_id);
    return;
  }
  DeviceAllocation devalloc = roots_[snode_tree_id];
  runtime_exec_->llvm_device()->dealloc_memory(devalloc);
  TI_DEBUG("SNode tree {} destroyed.", snode_tree_id);
}

}  // namespace taichi::lang
