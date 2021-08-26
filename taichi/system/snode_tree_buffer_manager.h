#pragma once
#include "taichi/llvm/llvm_context.h"
#include "taichi/inc/constants.h"
#include "taichi/struct/snode_tree.h"
#define TI_RUNTIME_HOST

#include <set>

using Ptr = uint8_t *;

TLANG_NAMESPACE_BEGIN

class LlvmProgramImpl;

class SNodeTreeBufferManager {
 public:
  SNodeTreeBufferManager(LlvmProgramImpl *prog);

  void merge_and_insert(Ptr ptr, std::size_t size);

  Ptr allocate(JITModule *runtime_jit,
               void *runtime,
               std::size_t size,
               std::size_t alignment,
               const int snode_tree_id,
               uint64 *result_buffer);

  void destroy(SNodeTree *snode_tree);

 private:
  std::set<std::pair<std::size_t, Ptr>> size_set_;
  std::map<Ptr, std::size_t> ptr_map_;
  LlvmProgramImpl *prog_;
  Ptr roots_[taichi_max_num_snode_trees];
  std::size_t sizes_[taichi_max_num_snode_trees];
};

TLANG_NAMESPACE_END
