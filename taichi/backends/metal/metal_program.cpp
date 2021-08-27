#include "metal_program.h"
#include "taichi/backends/metal/api.h"
#include "taichi/backends/metal/codegen_metal.h"
#include "taichi/backends/metal/struct_metal.h"

namespace taichi {
namespace lang {
MetalProgramImpl::MetalProgramImpl(CompileConfig &config_) {
  if (config_.arch == Arch::metal) {
    if (!metal::is_metal_api_available()) {
      TI_WARN("No Metal API detected.");
      config_.arch = host_arch();
    }
  }
  config = config_;
}

FunctionType MetalProgramImpl::compile_to_backend_executable(
    Kernel *kernel,
    OffloadedStmt *offloaded) {
  return metal::compile_to_metal_executable(kernel, metal_kernel_mgr_.get(),
                                            &metal_compiled_structs_.value(),
                                            offloaded);
}

std::size_t MetalProgramImpl::get_snode_num_dynamically_allocated(
    SNode *snode) {
  return metal_kernel_mgr_->get_snode_num_dynamically_allocated(snode);
}

void MetalProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                              uint64 **result_buffer_ptr,
                                              MemoryPool *memory_pool,
                                              KernelProfilerBase *profiler) {
  TI_ASSERT_INFO(config.use_llvm,
                 "Metal arch requires that LLVM being enabled");
  auto *const root = tree->root();

  metal_compiled_structs_ = metal::compile_structs(*root);
  if (metal_kernel_mgr_ == nullptr) {
    TI_ASSERT(*result_buffer_ptr == nullptr);
    *result_buffer_ptr = (uint64 *)memory_pool->allocate(
        sizeof(uint64) * taichi_result_buffer_entries, 8);
    metal::KernelManager::Params params;
    params.compiled_structs = metal_compiled_structs_.value();
    params.config = &config;
    params.mem_pool = memory_pool;
    params.host_result_buffer = *result_buffer_ptr;
    params.profiler = profiler;
    params.root_id = root->id;
    metal_kernel_mgr_ =
        std::make_unique<metal::KernelManager>(std::move(params));
  }
}

}  // namespace lang
}  // namespace taichi
