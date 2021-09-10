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
                                            &metal_compiled_structs_.value(),
                                            offloaded);
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
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  params_.mem_pool = memory_pool;
  params_.profiler = profiler;
}

void MetalProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    std::unordered_map<int, SNode *> &,
    SNodeGlobalVarExprMap &,
    uint64 *result_buffer) {
  // TODO: support materializing multiple snode trees
  TI_ASSERT_INFO(config->use_llvm,
                 "Metal arch requires that LLVM being enabled");
  auto *const root = tree->root();

  metal_compiled_structs_ = metal::compile_structs(*root);
  if (metal_kernel_mgr_ == nullptr) {
    params_.compiled_structs = metal_compiled_structs_.value();
    params_.config = config;
    params_.host_result_buffer = result_buffer;
    params_.root_id = root->id;
    metal_kernel_mgr_ =
        std::make_unique<metal::KernelManager>(std::move(params_));
  }
}

}  // namespace lang
}  // namespace taichi
