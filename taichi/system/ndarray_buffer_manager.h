#pragma once
#include "taichi/llvm/llvm_context.h"
#include "taichi/inc/constants.h"

#define TI_RUNTIME_HOST

#include <set>

using Ptr = uint8_t *;

TLANG_NAMESPACE_BEGIN

class LlvmProgramImpl;

class NdarrayBufferManager {
 public:
  NdarrayBufferManager(LlvmProgramImpl *prog);

  Ptr allocate(JITModule *runtime_jit,
               void *runtime,
               std::size_t size,
               std::size_t alignment,
               // const int snode_tree_id,
               uint64 *result_buffer);

 private:
  LlvmProgramImpl *prog_;
};

TLANG_NAMESPACE_END
