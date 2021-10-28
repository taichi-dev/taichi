#include "metal_program.h"
#include "taichi/backends/metal/codegen_metal.h"
#include "taichi/backends/metal/struct_metal.h"

namespace taichi {
namespace lang {
MetalProgramImpl::MetalProgramImpl(CompileConfig &config_)
    : ProgramImpl(config_) {
}

FunctionType MetalProgramImpl::compile(Kernel *kernel,
                                       OffloadedStmt *offloaded) {
  if (!kernel->lowered()) {
    kernel->lower();
  }
  return metal::compile_to_metal_executable(kernel, metal_kernel_mgr_.get(),
                                            &(compiled_runtime_module_.value()),
                                            compiled_snode_trees_, offloaded);
}

std::size_t MetalProgramImpl::get_snode_num_dynamically_allocated(SNode *snode,
                                                                  uint64 *) {
  // TODO: result_buffer is not used here since it's saved in params and already
  // available in metal_kernel_mgr
  return metal_kernel_mgr_->get_snode_num_dynamically_allocated(snode);
}

void MetalProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                           KernelProfilerBase *profiler,
                                           uint64 **result_buffer_ptr) {
  TI_ASSERT(*result_buffer_ptr == nullptr);
  TI_ASSERT(metal_kernel_mgr_ == nullptr);
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  compiled_runtime_module_ = metal::compile_runtime_module();

  metal::KernelManager::Params params;
  params.compiled_runtime_module = compiled_runtime_module_.value();
  params.config = config;
  params.host_result_buffer = *result_buffer_ptr;
  params.mem_pool = memory_pool;
  params.profiler = profiler;
  metal_kernel_mgr_ = std::make_unique<metal::KernelManager>(std::move(params));
}

void MetalProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    uint64 *result_buffer) {
  // TODO: support materializing multiple snode trees
  TI_ASSERT_INFO(config->use_llvm,
                 "Metal arch requires that LLVM being enabled");
  auto *const root = tree->root();
  auto csnode_tree = metal::compile_structs(*root);
  metal_kernel_mgr_->add_compiled_snode_tree(csnode_tree);
  compiled_snode_trees_.push_back(std::move(csnode_tree));
}

std::unique_ptr<AotModuleBuilder> MetalProgramImpl::make_aot_module_builder() {
  return std::make_unique<metal::AotModuleBuilderImpl>(
      &(compiled_runtime_module_.value()), compiled_snode_trees_,
      metal_kernel_mgr_->get_buffer_meta_data());
}

}  // namespace lang
}  // namespace taichi
