#ifdef TI_WITH_DX11

#include "taichi/runtime/program_impls/dx/dx_program.h"

#include "taichi/rhi/dx/dx_api.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/codegen/spirv/spirv_codegen.h"

namespace taichi::lang {
namespace directx11 {

FunctionType compile_to_executable(Kernel *kernel,
                                   gfx::GfxRuntime *runtime,
                                   const CompileConfig &compile_config,
                                   gfx::SNodeTreeManager *snode_tree_mgr) {
  auto handle = runtime->register_taichi_kernel(
      gfx::run_codegen(kernel, Arch::dx11, runtime->get_ti_device()->get_caps(),
                       snode_tree_mgr->get_compiled_structs(), compile_config));
  return [runtime, handle](RuntimeContext &ctx) {
    runtime->launch_kernel(handle, &ctx);
  };
}

}  // namespace directx11

Dx11ProgramImpl::Dx11ProgramImpl(CompileConfig &config) : ProgramImpl(config) {
}

FunctionType Dx11ProgramImpl::compile(const CompileConfig &compile_config,
                                      Kernel *kernel) {
  spirv::lower(compile_config, kernel);
  return directx11::compile_to_executable(
      kernel, runtime_.get(), compile_config, snode_tree_mgr_.get());
}

void Dx11ProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                          KernelProfilerBase *profiler,
                                          uint64 *&result_buffer_ptr,
                                          char *&device_arg_buffer_ptr) {
  result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

  device_ = directx11::make_dx11_device();

  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer_ptr;
  params.device = device_.get();
  runtime_ = std::make_unique<gfx::GfxRuntime>(std::move(params));
  snode_tree_mgr_ = std::make_unique<gfx::SNodeTreeManager>(runtime_.get());
}

void Dx11ProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  if (runtime_) {
    snode_tree_mgr_->materialize_snode_tree(tree);
  } else {
    gfx::CompiledSNodeStructs compiled_structs =
        gfx::compile_snode_structs(*tree->root());
    aot_compiled_snode_structs_.push_back(compiled_structs);
  }
}

void Dx11ProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                             uint64 *result_buffer) {
  snode_tree_mgr_->materialize_snode_tree(tree);
}

std::unique_ptr<AotModuleBuilder> Dx11ProgramImpl::make_aot_module_builder(
    const DeviceCapabilityConfig &caps) {
  if (runtime_) {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        snode_tree_mgr_->get_compiled_structs(), Arch::dx11, *config, caps);
  } else {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        aot_compiled_snode_structs_, Arch::dx11, *config, caps);
  }
}

DeviceAllocation Dx11ProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  return get_compute_device()->allocate_memory(
      {alloc_size, /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false});
}

}  // namespace taichi::lang

#endif
