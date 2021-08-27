#pragma once
#include "taichi/backends/metal/kernel_manager.h"
#include "taichi/backends/metal/struct_metal.h"
#include "taichi/system/memory_pool.h"
#include "taichi/common/logging.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/backends/metal/data_types.h"
#include "taichi/backends/metal/aot_module_builder_impl.h"
#include "taichi/backends/metal/struct_metal.h"

namespace taichi {
namespace lang {
class MetalProgramImpl {
 public:
  CompileConfig config;
  MetalProgramImpl(CompileConfig &config);
  FunctionType compile_to_backend_executable(Kernel *kernel,
                                             OffloadedStmt *offloaded);
  // TODO: materialize_runtime

  std::size_t get_snode_num_dynamically_allocated(SNode *snode);

  void materialize_snode_tree(
      SNodeTree *tree,
      uint64 **result_buffer_ptr,
      MemoryPool *memory_pool,
      KernelProfilerBase *profiler);

  void synchronize() {
    metal_kernel_mgr_->synchronize();
  }

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() {
    return std::make_unique<metal::AotModuleBuilderImpl>(
        &(metal_compiled_structs_.value()),
        metal_kernel_mgr_->get_buffer_meta_data());
  }

 private:
  std::optional<metal::CompiledStructs> metal_compiled_structs_;
  std::unique_ptr<metal::KernelManager> metal_kernel_mgr_;
};
}  // namespace lang
}  // namespace taichi
