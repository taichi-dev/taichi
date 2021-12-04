#include "taichi/codegen/spirv/spirv_codegen.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"

#include "taichi/backends/vulkan/runtime.h"

#include "taichi/backends/dx/dx_program.h"
#include "taichi/backends/dx/dx_api.h"

namespace taichi {
namespace lang {

namespace dx {

FunctionType compile_to_executable(Kernel *kernel, vulkan::VkRuntime *runtime) {
  auto handle = runtime->register_taichi_kernel(
      std::move(vulkan::run_codegen(kernel, runtime)));
  return [runtime, handle](RuntimeContext &ctx) {
    runtime->launch_kernel(handle, &ctx);
  };
}

} // namespace dx


FunctionType DxProgramImpl::compile(Kernel *kernel, OffloadedStmt *offloaded) {
  spirv::lower(kernel);
  return dx::compile_to_executable(kernel, runtime_.get());
}

void DxProgramImpl::compile_snode_tree_types(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees) {
  runtime_->materialize_snode_tree(tree);
}

void DxProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                        KernelProfilerBase *profiler,
                                        uint64 **result_buffer_ptr) {
  // 1. allocate result buffer
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  TI_ASSERT(dx::is_dx_api_available());

  device_ = dx::get_dx_device();

  vulkan::VkRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = device_.get();
  runtime_ = std::make_unique<vulkan::VkRuntime>(std::move(params));
}

void DxProgramImpl::materialize_snode_tree(
                         SNodeTree *tree,
                         std::vector<std::unique_ptr<SNodeTree>> &,
                         uint64 *result_buffer) {
  runtime_->materialize_snode_tree(tree);
}

DxProgramImpl::~DxProgramImpl() {
}

}  // namespace lang
}  // namespace taichi