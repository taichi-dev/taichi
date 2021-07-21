#pragma once
#include "taichi/system/unified_allocator.h"
#define TI_RUNTIME_HOST

#include <map>

TLANG_NAMESPACE_BEGIN

class Program;

class SNodeTreeBufferManager {
 public:
  std::map<int, std::unique_ptr<UnifiedAllocator>> allocators;
  Program *prog;

  SNodeTreeBufferManager(Program *prog);

  void *allocate(std::size_t size,
                 std::size_t alignment,
                 const int snode_tree_id);

  void destroy(const int snode_tree_id);
};

TLANG_NAMESPACE_END