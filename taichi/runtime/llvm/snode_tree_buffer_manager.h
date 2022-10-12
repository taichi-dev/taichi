#pragma once
#include "taichi/inc/constants.h"
#include "taichi/struct/snode_tree.h"
#define TI_RUNTIME_HOST

#include <set>

using Ptr = uint8_t *;

namespace taichi::lang {

class JITModule;
class LlvmRuntimeExecutor;

class SNodeTreeBufferManager {
 public:
  explicit SNodeTreeBufferManager(LlvmRuntimeExecutor *runtime_exec);

  void merge_and_insert(Ptr ptr, std::size_t size);

  Ptr allocate(JITModule *runtime_jit,
               void *runtime,
               std::size_t size,
               std::size_t alignment,
               const int snode_tree_id,
               uint64 *result_buffer);

  void destroy(SNodeTree *snode_tree);

  void save_root_buffer(const int save_id, const int snode_tree_id);

  void restore_root_buffer(const int save_id, const int snode_tree_id);

 private:
  std::set<std::pair<std::size_t, Ptr>> size_set_;
  std::map<Ptr, std::size_t> ptr_map_;
  LlvmRuntimeExecutor *runtime_exec_;
  Ptr roots_[kMaxNumSnodeTreesLlvm];
  std::size_t sizes_[kMaxNumSnodeTreesLlvm];
  // For autodiff checkpointing use
  std::map<int, std::vector<uint8_t>> root_buffer_snapshots_;
};

}  // namespace taichi::lang
