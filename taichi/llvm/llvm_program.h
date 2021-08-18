#pragma once
#include "taichi/system/snode_tree_buffer_manager.h"
#include "taichi/inc/constants.h"
#include "taichi/program/compile_config.h"
#include "taichi/common/logging.h"
#include "taichi/llvm/llvm_context.h"
#include "taichi/runtime/runtime.h"
#include "taichi/system/threading.h"

#include <memory>

namespace taichi {
namespace lang {
class LlvmProgramImpl {
 public:
  void *llvm_runtime{nullptr};
  std::unique_ptr<TaichiLLVMContext> llvm_context_host{nullptr};
  std::unique_ptr<TaichiLLVMContext> llvm_context_device{nullptr};
  std::unique_ptr<SNodeTreeBufferManager> snode_tree_buffer_manager{nullptr};
  std::unique_ptr<Runtime> runtime_mem_info{nullptr};
  std::unique_ptr<ThreadPool> thread_pool{nullptr};

  LlvmProgramImpl(CompileConfig &config);

  uint64 fetch_result_uint64(int i, uint64 *);

  void device_synchronize();

 private:
  CompileConfig config;
};
}  // namespace lang
}  // namespace taichi
