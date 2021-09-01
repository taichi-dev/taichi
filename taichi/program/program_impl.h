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
 protected:
  // TODO: To avoid divergence from Program....
  CompileConfig *config;

 public:
  ProgramImpl(CompileConfig &config);

  virtual FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) = 0;

  virtual void materialize_snode_tree(SNodeTree *tree,
                                      uint64 **result_buffer_ptr,
                                      MemoryPool *memory_pool,
                                      KernelProfilerBase *profiler) = 0;

  virtual std::size_t get_snode_num_dynamically_allocated(SNode *snode) = 0;

  virtual void synchronize() = 0;

  virtual std::unique_ptr<AotModuleBuilder> make_aot_module_builder() = 0;

  virtual ~ProgramImpl() {
  }

 private:
};

}  // namespace lang
}  // namespace taichi
