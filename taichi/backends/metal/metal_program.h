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
#include "taichi/program/program_impl.h"

namespace taichi {
namespace lang {
class MetalProgramImpl : public ProgramImpl {
 public:
  MetalProgramImpl(CompileConfig &config);
  FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;

  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override;

  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void materialize_snode_tree(
      SNodeTree *tree,
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      std::unordered_map<int, SNode *> &snodes,
      uint64 *result_buffer) override;

  void synchronize() override {
    metal_kernel_mgr_->synchronize();
  }

  virtual void destroy_snode_tree(SNodeTree *snode_tree) override{
      TI_NOT_IMPLEMENTED}

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override {
    return std::make_unique<metal::AotModuleBuilderImpl>(
        &(metal_compiled_structs_.value()),
        metal_kernel_mgr_->get_buffer_meta_data());
  }
  ~MetalProgramImpl() {
  }

 private:
  std::optional<metal::CompiledStructs> metal_compiled_structs_;
  std::unique_ptr<metal::KernelManager> metal_kernel_mgr_;
  metal::KernelManager::Params params_;
};
}  // namespace lang
}  // namespace taichi
