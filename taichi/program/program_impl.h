#pragma once
#include "taichi/system/memory_pool.h"
#include "taichi/common/logging.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/program/aot_module_builder.h"

namespace taichi {
namespace lang {
class ProgramImpl {
 public:
  // TODO: Make it safer, we exposed it for now as it's directly accessed
  // outside.
  CompileConfig *config;

 public:
  ProgramImpl(CompileConfig &config);

  /**
   * Codegen to specific backend
   */
  virtual FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) = 0;

  /**
   * Allocate runtime buffer, e.g result_buffer or backend specific runtime
   * buffer, e.g. preallocated_device_buffer on CUDA.
   */
  virtual void materialize_runtime(MemoryPool *memory_pool,
                                   KernelProfilerBase *profiler,
                                   uint64 **result_buffer_ptr) = 0;

  /**
   * Run StructCompiler for the backend.
   */
  virtual void materialize_snode_tree(
      SNodeTree *tree,
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      std::unordered_map<int, SNode *> &snodes,
      uint64 *result_buffer_ptr) = 0;

  virtual void destroy_snode_tree(SNodeTree *snode_tree) = 0;

  virtual std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) = 0;

  /**
   * Perform a backend synchronization.
   */
  virtual void synchronize() = 0;

  /**
   * Make a AotModulerBuilder, currently only supported by metal and wasm.
   */
  virtual std::unique_ptr<AotModuleBuilder> make_aot_module_builder() = 0;

  virtual ~ProgramImpl() {
  }

 private:
};

}  // namespace lang
}  // namespace taichi
