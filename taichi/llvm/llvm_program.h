#pragma once
#include "taichi/system/snode_tree_buffer_manager.h"
#include "taichi/inc/constants.h"
#include "taichi/program/compile_config.h"
#include "taichi/common/logging.h"
#include "taichi/llvm/llvm_context.h"

#include <memory>

namespace taichi {
namespace lang {
class LlvmProgramImpl {
 public:
  std::unique_ptr<TaichiLLVMContext> llvm_context_host{nullptr};
  std::unique_ptr<SNodeTreeBufferManager> snode_tree_buffer_manager{nullptr};

  LlvmProgramImpl(CompileConfig config);

  uint64 fetch_result_uint64(int i, uint64 *);

  void device_synchronize();

 private:
  CompileConfig config;
};
}  // namespace lang
}  // namespace taichi
