#include "taichi/codegen/spirv/spirv_codegen.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"

#include "taichi/backends/vulkan/runtime.h"

#include "taichi/backends/dx/dx_program.h"
#include "taichi/backends/dx/dx_api.h"
#include "taichi/backends/dx/dx_device.h"

namespace taichi {
namespace lang {
namespace directx11 {
FunctionType compile_to_executable(Kernel *kernel, vulkan::VkRuntime *runtime) {
  auto handle = runtime->register_taichi_kernel(
      std::move(vulkan::run_codegen(kernel, runtime)));
  return [runtime, handle](RuntimeContext &ctx) {
    runtime->launch_kernel(handle, &ctx);
  };
}
}  // namespace directx11

FunctionType Dx11ProgramImpl::compile(Kernel *kernel,
                                      OffloadedStmt *offloaded) {
  spirv::lower(kernel);
  return directx11::compile_to_executable(kernel, runtime_.get());
}

void Dx11ProgramImpl::compile_snode_tree_types(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees) {
  runtime_->materialize_snode_tree(tree);
}

void Dx11ProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                          KernelProfilerBase *profiler,
                                          uint64 **result_buffer_ptr) {
  // 1. allocate result buffer
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  TI_ASSERT(directx11::is_dx_api_available());

  device_ = std::make_unique<directx11::DxDevice>();

  vulkan::VkRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = device_.get();
  runtime_ = std::make_unique<vulkan::VkRuntime>(std::move(params));
}

void Dx11ProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    uint64 *result_buffer) {
  runtime_->materialize_snode_tree(tree);
}

Dx11ProgramImpl::~Dx11ProgramImpl() {
}

}  // namespace lang
}  // namespace taichi