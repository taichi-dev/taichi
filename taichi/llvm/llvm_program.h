#pragma once
#include "taichi/system/snode_tree_buffer_manager.h"
#include "taichi/inc/constants.h"
#include "taichi/program/compile_config.h"
#include "taichi/common/logging.h"
#include "taichi/llvm/llvm_context.h"
#include "taichi/runtime/runtime.h"
#include "taichi/system/threading.h"
#include "llvm/IR/Module.h"
#include "taichi/struct/struct.h"
#include "taichi/struct/struct_llvm.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/system/memory_pool.h"
#include "taichi/program/program_impl.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

#include <memory>

namespace taichi {
namespace lang {
class StructCompiler;

class LlvmProgramImpl : public ProgramImpl {
 public:
  LlvmProgramImpl(CompileConfig &config, KernelProfilerBase *profiler);

  void initialize_host();

  /**
   * Initializes Program#llvm_context_device, if this has not been done.
   *
   * Not thread safe.
   */
  void maybe_initialize_cuda_llvm_context();

  TaichiLLVMContext *get_llvm_context(Arch arch) {
    if (arch_is_cpu(arch)) {
      return llvm_context_host.get();
    } else {
      return llvm_context_device.get();
    }
  }

  LLVMRuntime *get_llvm_runtime() {
    return static_cast<LLVMRuntime *>(llvm_runtime);
  }

  FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;

  void materialize_snode_tree(
      SNodeTree *tree,
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      std::unordered_map<int, SNode *> &snodes,
      SNodeGlobalVarExprMap &snode_to_glb_var_exprs_,
      uint64 *result_buffer) override;

  template <typename T>
  T fetch_result(int i, uint64 *result_buffer) {
    return taichi_union_cast_with_different_sizes<T>(
        fetch_result_uint64(i, result_buffer));
  }

  /**
   * Initializes the runtime system for LLVM based backends.
   */
  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override;

  void destroy_snode_tree(SNodeTree *snode_tree) {
    snode_tree_buffer_manager->destroy(snode_tree);
  }

  void print_memory_profiler_info(
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      uint64 *result_buffer);

  void synchronize() override;

  void check_runtime_error(uint64 *result_buffer);

  void finalize();

 private:
  std::unique_ptr<llvm::Module> clone_struct_compiler_initial_context(
      const std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      TaichiLLVMContext *tlctx);

  /**
   * Initializes the SNodes for LLVM based backends.
   */
  void initialize_llvm_runtime_snodes(const SNodeTree *tree,
                                      StructCompiler *scomp,
                                      uint64 *result_buffer);

  /**
   * Sets the attributes of the Exprs that are backed by SNodes.
   */
  void materialize_snode_expr_attributes(
      SNodeGlobalVarExprMap &snode_to_glb_var_exprs_);

  uint64 fetch_result_uint64(int i, uint64 *result_buffer);

  template <typename T, typename... Args>
  T runtime_query(const std::string &key, uint64 *result_buffer, Args... args) {
    TI_ASSERT(arch_uses_llvm(config->arch));

    TaichiLLVMContext *tlctx = nullptr;
    if (llvm_context_device) {
      tlctx = llvm_context_device.get();
    } else {
      tlctx = llvm_context_host.get();
    }

    auto runtime = tlctx->runtime_jit_module;
    runtime->call<void *, Args...>("runtime_" + key, llvm_runtime,
                                   std::forward<Args>(args)...);
    return taichi_union_cast_with_different_sizes<T>(fetch_result_uint64(
        taichi_result_buffer_runtime_query_id, result_buffer));
  }

  void print_list_manager_info(void *list_manager, uint64 *result_buffer);

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override {
    TI_NOT_IMPLEMENTED;
  }

 private:
  std::unique_ptr<TaichiLLVMContext> llvm_context_host{nullptr};
  std::unique_ptr<TaichiLLVMContext> llvm_context_device{nullptr};
  std::unique_ptr<ThreadPool> thread_pool{nullptr};
  std::unique_ptr<Runtime> runtime_mem_info{nullptr};
  std::unique_ptr<SNodeTreeBufferManager> snode_tree_buffer_manager{nullptr};
  void *llvm_runtime{nullptr};
  void *preallocated_device_buffer{nullptr};  // TODO: move to memory allocator
};
}  // namespace lang
}  // namespace taichi
