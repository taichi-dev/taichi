#include "taichi/backends/dx/dx_program.h"
#include "taichi/backends/dx/dx_device.h"

namespace taichi {
namespace lang {
namespace directx11 {

FunctionType compile_to_executable(Kernel *kernel, vulkan::VkRuntime *runtime) {
  auto handle = runtime->register_taichi_kernel(std::move(vulkan::run_codegen(
      kernel, runtime->get_ti_device(), runtime->get_compiled_structs())));
  return [runtime, handle](RuntimeContext &ctx) {
    runtime->launch_kernel(handle, &ctx);
  };
}

}  // namespace directx11

Dx11ProgramImpl::Dx11ProgramImpl(CompileConfig &config) : ProgramImpl(config) {
}

FunctionType Dx11ProgramImpl::compile(Kernel *kernel,
                                      OffloadedStmt *offloaded) {
  spirv::lower(kernel);
  return directx11::compile_to_executable(kernel, runtime_.get());
}

void Dx11ProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                          KernelProfilerBase *profiler,
                                          uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

  device_ = std::make_unique<directx11::Dx11Device>();

  vulkan::VkRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = device_.get();
  runtime_ = std::make_unique<vulkan::VkRuntime>(std::move(params));
}

void Dx11ProgramImpl::synchronize() {
  TI_NOT_IMPLEMENTED;
}

void Dx11ProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
    uint64 *result_buffer_ptr) {
  runtime_->materialize_snode_tree(tree);
}

std::unique_ptr<AotModuleBuilder> Dx11ProgramImpl::make_aot_module_builder() {
  return nullptr;
}

void Dx11ProgramImpl::destroy_snode_tree(SNodeTree *snode_tree) {
  TI_NOT_IMPLEMENTED;
}

}  // namespace lang
}  // namespace taichi
