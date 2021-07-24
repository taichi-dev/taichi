#pragma once
#include "taichi/llvm/llvm_context.h"
#include "taichi/inc/constants.h"
#define TI_RUNTIME_HOST

#include <set>

using Ptr = uint8_t *;

TLANG_NAMESPACE_BEGIN

class Program;

class SNodeTreeBufferManager {
 public:
  SNodeTreeBufferManager(Program *prog);

  Ptr allocate(JITModule *runtime_jit,
               void *runtime,
               std::size_t size,
               std::size_t alignment,
               const int snode_tree_id);

  void destroy(const int snode_tree_id);

 private:
  std::set<std::pair<std::size_t, Ptr>> size_set;
  std::map<Ptr, std::size_t> Ptr_map;
  Program *prog;
  Ptr roots[taichi_max_num_snode_trees];
  std::size_t sizes[taichi_max_num_snode_trees];
};

TLANG_NAMESPACE_END
